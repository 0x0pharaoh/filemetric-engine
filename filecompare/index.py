"""
filecompare/index.py - FileIndex and DynamicFileIndex
"""

from __future__ import annotations

import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .cache import VectorCache
from .types import FileMatch, MultiResult


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_cached(path, cache):
    path = Path(path)
    file_hash = VectorCache.hash_file(path)
    if cache is not None:
        hit = cache.get(file_hash)
        if hit is not None:
            return file_hash, hit[0]
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = _clean(raw)
    if cache is not None:
        cache.set(file_hash, cleaned)
    return file_hash, cleaned


class FileIndex:
    """
    Immutable TF-IDF index. Build once, query many times.
    Each query is a single sparse matrix-vector multiply.
    """

    def __init__(self, paths, vectorizer, matrix, built_at):
        self.paths = paths
        self._vectorizer = vectorizer
        self._matrix = matrix
        self.built_at = built_at
        self.size = len(paths)

    @classmethod
    def build(cls, base_files, cache=None, max_workers=8, vectorizer_kwargs=None, verbose=False):
        """Read, clean, and vectorise all base_files into an index."""
        if not base_files:
            raise ValueError("base_files cannot be empty.")

        paths = [str(p) for p in base_files]
        if verbose:
            print(f"[FileIndex] Reading {len(paths)} files ({max_workers} workers)...")

        texts = [None] * len(paths)

        def _load(idx, path):
            _, text = _read_cached(path, cache)
            return idx, text

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load, i, p): i for i, p in enumerate(paths)}
            done = 0
            for future in as_completed(futures):
                idx, text = future.result()
                texts[idx] = text
                done += 1
                if verbose and done % 500 == 0:
                    print(f"[FileIndex]   {done}/{len(paths)} files read...")

        if verbose:
            print("[FileIndex] Fitting TF-IDF vectorizer...")

        vkw = {"analyzer": "word", "ngram_range": (1, 2), "sublinear_tf": True,
               "min_df": 1, "max_features": 200_000}
        if vectorizer_kwargs:
            vkw.update(vectorizer_kwargs)

        vectorizer = TfidfVectorizer(**vkw)
        matrix = vectorizer.fit_transform(texts)

        if verbose:
            print(f"[FileIndex] Done. {matrix.shape[0]} docs x {matrix.shape[1]} features")

        return cls(paths=paths, vectorizer=vectorizer, matrix=matrix, built_at=time.time())

    def save(self, path):
        """Pickle the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"paths": self.paths, "vectorizer": self._vectorizer,
                 "matrix": self._matrix, "built_at": self.built_at},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path):
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(paths=data["paths"], vectorizer=data["vectorizer"],
                   matrix=data["matrix"], built_at=data["built_at"])

    def query(self, main_file, cache=None, top_n=None, threshold=0.0, sort=True):
        """Compare main_file against every file in the index."""
        _, text = _read_cached(main_file, cache)
        query_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        matches = [
            FileMatch(file=self.paths[i], percentage=round(float(scores[i]) * 100, 2))
            for i in range(len(self.paths))
            if round(float(scores[i]) * 100, 2) >= threshold
        ]
        if sort:
            matches.sort(key=lambda m: m.percentage, reverse=True)
        if top_n is not None:
            matches = matches[:top_n]
        return MultiResult(file=str(main_file), compare=matches)

    def query_text(self, text, top_n=None, threshold=0.0, sort=True):
        """Compare raw text against every file in the index."""
        cleaned = _clean(text)
        query_vec = self._vectorizer.transform([cleaned])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        matches = [
            FileMatch(file=self.paths[i], percentage=round(float(scores[i]) * 100, 2))
            for i in range(len(self.paths))
            if round(float(scores[i]) * 100, 2) >= threshold
        ]
        if sort:
            matches.sort(key=lambda m: m.percentage, reverse=True)
        if top_n is not None:
            matches = matches[:top_n]
        return MultiResult(file="<text_input>", compare=matches)

    def add_files(self, new_files, cache=None, max_workers=8):
        """Return a NEW FileIndex with additional files. Re-fits on all docs."""
        all_paths = self.paths + [str(p) for p in new_files]
        return FileIndex.build(all_paths, cache=cache, max_workers=max_workers)

    def info(self):
        import datetime
        return {
            "files_indexed": self.size,
            "vocab_size": len(self._vectorizer.vocabulary_),
            "matrix_shape": list(self._matrix.shape),
            "built_at": datetime.datetime.fromtimestamp(self.built_at).isoformat(),
        }


class DynamicFileIndex:
    """
    A FileIndex wrapper for incremental file adds.

    New files are staged in a pending buffer (O(1) add).
    When buffer reaches merge_threshold, a full rebuild is triggered.
    Queries check both the main index and the pending buffer.

    Usage::

        idx = DynamicFileIndex(merge_threshold=50, cache=cache)
        idx.build_initial(existing_files)
        idx.save("groups/legal.dyn")

        # On every upload:
        idx = DynamicFileIndex.load("groups/legal.dyn")
        idx.add_file("uploads/new_doc.txt")
        idx.save("groups/legal.dyn")

        result = idx.query("compare_this.txt", top_n=10)
    """

    def __init__(self, merge_threshold=50, cache=None, max_workers=8):
        self.merge_threshold = merge_threshold
        self.cache = cache
        self.max_workers = max_workers
        self._index: Optional[FileIndex] = None
        self._pending: list[str] = []

    def build_initial(self, base_files, verbose=False):
        """Seed the index from a bulk set of files. Clears existing state."""
        if base_files:
            self._index = FileIndex.build(
                base_files, cache=self.cache,
                max_workers=self.max_workers, verbose=verbose,
            )
        self._pending = []

    def add_file(self, path, force_merge=False):
        """
        Add one file to the pending buffer.
        Auto-merges when buffer hits merge_threshold.
        Returns True if a merge was triggered.
        """
        self._pending.append(str(path))
        merged = False
        if force_merge or len(self._pending) >= self.merge_threshold:
            self._merge()
            merged = True
        return merged

    def add_files(self, paths, force_merge=False):
        """
        Add multiple files at once.
        Single merge check at the end, single disk write.
        Returns True if a merge was triggered.
        """
        self._pending.extend(str(p) for p in paths)
        merged = False
        if force_merge or len(self._pending) >= self.merge_threshold:
            self._merge()
            merged = True
        return merged

    def force_merge(self):
        """Force an immediate rebuild regardless of buffer size."""
        if self._pending:
            self._merge()

    def _merge(self):
        existing = self._index.paths if self._index else []
        all_paths = existing + self._pending
        self._index = FileIndex.build(
            all_paths, cache=self.cache, max_workers=self.max_workers,
        )
        self._pending = []

    def remove_file(self, path):
        """
        Remove a file from the index.
        Pending buffer: O(1) pop.
        Main index: triggers a full rebuild.
        Returns True if found and removed, False if not found.
        """
        path_str = str(path)

        if path_str in self._pending:
            self._pending.remove(path_str)
            return True

        if self._index and path_str in self._index.paths:
            remaining = [p for p in self._index.paths if p != path_str]
            self._index = FileIndex.build(
                remaining, cache=self.cache, max_workers=self.max_workers,
            ) if remaining else None
            return True

        return False

    def query(self, main_file, top_n=None, threshold=0.0, sort=True):
        """Query against both main index and pending buffer, results merged."""
        if self._index is None and not self._pending:
            raise RuntimeError("Index is empty. Call build_initial() or add_file() first.")

        all_matches: list[FileMatch] = []

        if self._index is not None:
            result = self._index.query(
                main_file, cache=self.cache, threshold=threshold, sort=False,
            )
            all_matches.extend(result.compare)

        if self._pending:
            from .compare import compare_one_to_many
            buf_result = compare_one_to_many(
                main_file, self._pending, cache=self.cache,
                threshold=threshold, sort=False,
            )
            all_matches.extend(buf_result.compare)

        if sort:
            all_matches.sort(key=lambda m: m.percentage, reverse=True)
        if top_n is not None:
            all_matches = all_matches[:top_n]

        return MultiResult(file=str(main_file), compare=all_matches)

    def query_text(self, text, top_n=None, threshold=0.0, sort=True):
        """Query raw text against both layers."""
        from .compare import _clean, _pct

        all_matches: list[FileMatch] = []

        if self._index is not None:
            result = self._index.query_text(text, threshold=threshold, sort=False)
            all_matches.extend(result.compare)

        if self._pending:
            pending_texts = [_read_cached(p, self.cache)[1] for p in self._pending]
            cleaned_query = _clean(text)
            vec = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1,
            )
            all_texts = [cleaned_query] + pending_texts
            mat = vec.fit_transform(all_texts)
            scores = cosine_similarity(mat[0:1], mat[1:])[0]
            for i, score in enumerate(scores):
                pct = _pct(score)
                if pct >= threshold:
                    all_matches.append(FileMatch(file=self._pending[i], percentage=pct))

        if sort:
            all_matches.sort(key=lambda m: m.percentage, reverse=True)
        if top_n:
            all_matches = all_matches[:top_n]
        return MultiResult(file="<text_input>", compare=all_matches)

    def save(self, path):
        """Persist main index and pending buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"index": self._index, "pending": self._pending,
                 "merge_threshold": self.merge_threshold},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path, cache=None, max_workers=8):
        """Load a persisted DynamicFileIndex from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(merge_threshold=data["merge_threshold"], cache=cache, max_workers=max_workers)
        obj._index = data["index"]
        obj._pending = data["pending"]
        return obj

    def info(self):
        import datetime
        main_count = self._index.size if self._index else 0
        built_at = (
            datetime.datetime.fromtimestamp(self._index.built_at).isoformat()
            if self._index else None
        )
        return {
            "files_in_main_index": main_count,
            "files_in_pending_buffer": len(self._pending),
            "total_files": main_count + len(self._pending),
            "merge_threshold": self.merge_threshold,
            "main_index_built_at": built_at,
            "pending_files": self._pending,
        }
