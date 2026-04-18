"""
tests/test_filecompare.py
-------------------------
Tests for all three layers: compare functions, FileIndex, and VectorCache.
Run with: pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from filemetric_engine import (
    FileIndex,
    MultiResult,
    PairResult,
    VectorCache,
    compare_files,
    compare_one_to_many,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEXTS = {
    "main": (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning is transforming industries. "
        "Python is widely used for data science."
    ),
    "very_similar": (
        "The quick brown fox leaps over a sleepy dog. "
        "Machine learning and deep learning are transforming many sectors. "
        "Python remains the top language for data science."
    ),
    "somewhat_similar": (
        "Python is good for automation tasks. "
        "Data science uses Python heavily. "
        "The fox is quick and brown."
    ),
    "unrelated": (
        "Natural language processing enables computers to understand text. "
        "Large language models are trained on vast corpora. "
        "Transformers revolutionised NLP research."
    ),
}


@pytest.fixture
def tmp_files(tmp_path):
    """Write sample texts to temp files and return a dict of {name: Path}."""
    paths = {}
    for name, content in TEXTS.items():
        p = tmp_path / f"{name}.txt"
        p.write_text(content)
        paths[name] = p
    return paths


@pytest.fixture
def tmp_cache(tmp_path):
    """Return a VectorCache backed by a temp DB."""
    return VectorCache(db_path=tmp_path / "test_cache.db")


# ---------------------------------------------------------------------------
# PairResult / compare_files
# ---------------------------------------------------------------------------

class TestCompareFiles:
    def test_returns_pair_result(self, tmp_files):
        result = compare_files(tmp_files["main"], tmp_files["very_similar"])
        assert isinstance(result, PairResult)

    def test_percentage_range(self, tmp_files):
        result = compare_files(tmp_files["main"], tmp_files["very_similar"])
        assert 0.0 <= result.common_in_percentage <= 100.0

    def test_identical_files_near_100(self, tmp_files):
        result = compare_files(tmp_files["main"], tmp_files["main"])
        assert result.common_in_percentage == pytest.approx(100.0, abs=0.1)

    def test_unrelated_files_near_0(self, tmp_files):
        result = compare_files(tmp_files["main"], tmp_files["unrelated"])
        assert result.common_in_percentage < 10.0

    def test_similar_higher_than_unrelated(self, tmp_files):
        similar = compare_files(tmp_files["main"], tmp_files["very_similar"])
        unrelated = compare_files(tmp_files["main"], tmp_files["unrelated"])
        assert similar.common_in_percentage > unrelated.common_in_percentage

    def test_to_dict_shape(self, tmp_files):
        result = compare_files(tmp_files["main"], tmp_files["very_similar"])
        d = result.to_dict()
        assert set(d.keys()) == {"file_1", "file_2", "common_in_percentage"}
        assert isinstance(d["common_in_percentage"], float)

    def test_with_cache(self, tmp_files, tmp_cache):
        # First call — cache miss
        r1 = compare_files(tmp_files["main"], tmp_files["very_similar"], cache=tmp_cache)
        # Second call — cache hit
        r2 = compare_files(tmp_files["main"], tmp_files["very_similar"], cache=tmp_cache)
        assert r1.common_in_percentage == r2.common_in_percentage


# ---------------------------------------------------------------------------
# MultiResult / compare_one_to_many
# ---------------------------------------------------------------------------

class TestCompareOneToMany:
    def test_returns_multi_result(self, tmp_files):
        result = compare_one_to_many(
            tmp_files["main"],
            [tmp_files["very_similar"], tmp_files["unrelated"]],
        )
        assert isinstance(result, MultiResult)

    def test_correct_number_of_comparisons(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        result = compare_one_to_many(tmp_files["main"], base)
        assert len(result.compare) == 3

    def test_sorted_descending(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        result = compare_one_to_many(tmp_files["main"], base, sort=True)
        pcts = [m.percentage for m in result.compare]
        assert pcts == sorted(pcts, reverse=True)

    def test_top_n_filter(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        result = compare_one_to_many(tmp_files["main"], base, top_n=2)
        assert len(result.compare) == 2

    def test_threshold_filter(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        result = compare_one_to_many(tmp_files["main"], base, threshold=5.0)
        for m in result.compare:
            assert m.percentage >= 5.0

    def test_to_dict_shape(self, tmp_files):
        result = compare_one_to_many(
            tmp_files["main"],
            [tmp_files["very_similar"], tmp_files["unrelated"]],
        )
        d = result.to_dict()
        assert "file" in d
        assert "compare" in d
        for entry in d["compare"]:
            assert "file" in entry
            assert "percentage" in entry

    def test_empty_base_raises(self, tmp_files):
        with pytest.raises(ValueError):
            compare_one_to_many(tmp_files["main"], [])


# ---------------------------------------------------------------------------
# VectorCache
# ---------------------------------------------------------------------------

class TestVectorCache:
    def test_set_and_get(self, tmp_cache):
        tmp_cache.set("abc123", "hello world", None)
        result = tmp_cache.get("abc123")
        assert result is not None
        assert result[0] == "hello world"

    def test_miss_returns_none(self, tmp_cache):
        assert tmp_cache.get("nonexistent_hash") is None

    def test_has(self, tmp_cache):
        tmp_cache.set("xyz", "text")
        assert tmp_cache.has("xyz") is True
        assert tmp_cache.has("missing") is False

    def test_invalidate(self, tmp_cache):
        tmp_cache.set("todelete", "text")
        tmp_cache.invalidate("todelete")
        assert tmp_cache.has("todelete") is False

    def test_clear(self, tmp_cache):
        tmp_cache.set("a", "text a")
        tmp_cache.set("b", "text b")
        tmp_cache.clear()
        assert tmp_cache.stats()["entries"] == 0

    def test_stats(self, tmp_cache):
        tmp_cache.set("h1", "some text")
        stats = tmp_cache.stats()
        assert stats["entries"] == 1
        assert "db_path" in stats

    def test_hash_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello")
        h1 = VectorCache.hash_file(p)
        h2 = VectorCache.hash_file(p)
        assert h1 == h2  # deterministic

        p.write_text("hello changed")
        h3 = VectorCache.hash_file(p)
        assert h1 != h3  # changed content → different hash


# ---------------------------------------------------------------------------
# FileIndex
# ---------------------------------------------------------------------------

class TestFileIndex:
    def test_build_and_query(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        idx = FileIndex.build(base)
        result = idx.query(tmp_files["main"])
        assert isinstance(result, MultiResult)
        assert len(result.compare) == 3

    def test_query_sorted(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        idx = FileIndex.build(base)
        result = idx.query(tmp_files["main"], sort=True)
        pcts = [m.percentage for m in result.compare]
        assert pcts == sorted(pcts, reverse=True)

    def test_query_top_n(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        idx = FileIndex.build(base)
        result = idx.query(tmp_files["main"], top_n=2)
        assert len(result.compare) == 2

    def test_query_threshold(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"], tmp_files["unrelated"]]
        idx = FileIndex.build(base)
        result = idx.query(tmp_files["main"], threshold=5.0)
        for m in result.compare:
            assert m.percentage >= 5.0

    def test_query_text(self, tmp_files):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"]]
        idx = FileIndex.build(base)
        result = idx.query_text("Python is used for data science", top_n=2)
        assert len(result.compare) <= 2

    def test_save_and_load(self, tmp_files, tmp_path):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"]]
        idx = FileIndex.build(base)

        index_path = tmp_path / "test.pkl"
        idx.save(index_path)

        loaded = FileIndex.load(index_path)
        result = loaded.query(tmp_files["main"])
        assert len(result.compare) == 2

    def test_save_load_same_results(self, tmp_files, tmp_path):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"]]
        idx = FileIndex.build(base)
        r1 = idx.query(tmp_files["main"])

        idx.save(tmp_path / "idx.pkl")
        idx2 = FileIndex.load(tmp_path / "idx.pkl")
        r2 = idx2.query(tmp_files["main"])

        assert len(r1.compare) == len(r2.compare)
        for m1, m2 in zip(r1.compare, r2.compare):
            assert m1.file == m2.file
            assert m1.percentage == pytest.approx(m2.percentage, abs=0.01)

    def test_info(self, tmp_files):
        idx = FileIndex.build([tmp_files["very_similar"]])
        info = idx.info()
        assert "files_indexed" in info
        assert info["files_indexed"] == 1

    def test_build_with_cache(self, tmp_files, tmp_cache):
        base = [tmp_files["very_similar"], tmp_files["somewhat_similar"]]
        idx1 = FileIndex.build(base, cache=tmp_cache)
        # Second build should hit cache
        idx2 = FileIndex.build(base, cache=tmp_cache)
        assert idx1.size == idx2.size

    def test_add_files(self, tmp_files):
        base = [tmp_files["very_similar"]]
        idx = FileIndex.build(base)
        assert idx.size == 1

        idx2 = idx.add_files([tmp_files["unrelated"]])
        assert idx2.size == 2
