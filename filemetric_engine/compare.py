"""
filemetric_engine/compare.py
----------------------
Simple functional API — no need to manage a FileIndex manually.

Use these when:
  - You have a small number of files (< a few hundred)
  - You don't need to reuse the index across multiple queries
  - You want the simplest possible interface

For repeated queries against the same large set of base files,
use FileIndex directly (see index.py) — it's dramatically faster.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .cache import VectorCache
from .types import FileMatch, MultiResult, PairResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _read_cached(path: str | Path, cache: Optional[VectorCache]) -> tuple[str, str]:
    path = Path(path)
    file_hash = VectorCache.hash_file(path)
    if cache:
        hit = cache.get(file_hash)
        if hit:
            return file_hash, hit[0]
    cleaned = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    if cache:
        cache.set(file_hash, cleaned)
    return file_hash, cleaned


def _pct(score: float) -> float:
    return round(float(np.clip(score, 0.0, 1.0)) * 100, 2)


def _tfidf_matrix(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1
    )
    return cosine_similarity(vec.fit_transform(texts))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compare_files(
    file_1: str | Path,
    file_2: str | Path,
    cache: Optional[VectorCache] = None,
) -> PairResult:
    """
    Compare two files and return their similarity percentage.

    Parameters
    ----------
    file_1  : Main file path.
    file_2  : Base file path.
    cache   : Optional VectorCache to avoid re-reading unchanged files.

    Returns
    -------
    PairResult
        {"file_1": "...", "file_2": "...", "common_in_percentage": 42.5}

    Example
    -------
    >>> from filemetric_engine import compare_files, VectorCache
    >>> cache = VectorCache()
    >>> result = compare_files("doc_a.txt", "doc_b.txt", cache=cache)
    >>> print(result.common_in_percentage)
    67.34
    """
    _, text_1 = _read_cached(file_1, cache)
    _, text_2 = _read_cached(file_2, cache)
    matrix = _tfidf_matrix([text_1, text_2])
    return PairResult(
        file_1=str(file_1),
        file_2=str(file_2),
        common_in_percentage=_pct(matrix[0][1]),
    )


def compare_one_to_many(
    main_file: str | Path,
    base_files: List[str | Path],
    cache: Optional[VectorCache] = None,
    top_n: Optional[int] = None,
    threshold: float = 0.0,
    sort: bool = True,
) -> MultiResult:
    """
    Compare one main file against a list of base files.

    For large base_files lists (hundreds+), prefer building a FileIndex once
    and reusing it, rather than calling this function repeatedly.

    Parameters
    ----------
    main_file  : File to compare.
    base_files : List of files to compare against.
    cache      : Optional VectorCache.
    top_n      : Return only top-n results.
    threshold  : Exclude results below this percentage (0–100).
    sort       : Sort results highest → lowest.

    Returns
    -------
    MultiResult
        {
            "file": "main.txt",
            "compare": [
                {"file": "base_a.txt", "percentage": 82.1},
                {"file": "base_b.txt", "percentage": 34.5},
            ]
        }

    Example
    -------
    >>> from filemetric_engine import compare_one_to_many, VectorCache
    >>> cache = VectorCache()
    >>> result = compare_one_to_many("new.txt", ["ref1.txt", "ref2.txt"], cache=cache)
    >>> for m in result.compare:
    ...     print(m.file, m.percentage)
    """
    if not base_files:
        raise ValueError("base_files cannot be empty.")

    all_paths = [main_file] + list(base_files)
    all_texts = [_read_cached(p, cache)[1] for p in all_paths]

    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1
    )
    matrix = vec.fit_transform(all_texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:])[0]  # 1 × n_base

    matches = [
        FileMatch(file=str(base_files[i]), percentage=_pct(scores[i]))
        for i in range(len(base_files))
        if _pct(scores[i]) >= threshold
    ]

    if sort:
        matches.sort(key=lambda m: m.percentage, reverse=True)
    if top_n:
        matches = matches[:top_n]

    return MultiResult(file=str(main_file), compare=matches)
