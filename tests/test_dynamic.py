"""
tests/test_dynamic.py
---------------------
Tests for DynamicFileIndex and IndexRegistry.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
from filemetric_engine import DynamicFileIndex, IndexRegistry, VectorCache, GroupNotFoundError


TEXTS = {
    "main":    "The quick brown fox jumps over the lazy dog. Python is used for data science.",
    "sim_a":   "The quick brown fox leaps over a sleepy dog. Python remains top for data science.",
    "sim_b":   "Python is good for automation. Data science uses Python heavily.",
    "unrela":  "Natural language processing enables computers to understand text.",
    "new_1":   "Machine learning transforms industries. Deep learning advances AI.",
    "new_2":   "Neural networks process images. Convolutional nets dominate vision tasks.",
}


@pytest.fixture
def tmp_files(tmp_path):
    paths = {}
    for name, content in TEXTS.items():
        p = tmp_path / f"{name}.txt"
        p.write_text(content)
        paths[name] = p
    return paths


@pytest.fixture
def tmp_cache(tmp_path):
    return VectorCache(db_path=tmp_path / "cache.db")


# ---------------------------------------------------------------------------
# DynamicFileIndex
# ---------------------------------------------------------------------------

class TestDynamicFileIndex:

    def test_build_initial_and_query(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=10)
        idx.build_initial([tmp_files["sim_a"], tmp_files["sim_b"], tmp_files["unrela"]])
        result = idx.query(tmp_files["main"])
        assert len(result.compare) == 3

    def test_add_file_goes_to_pending(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=10)
        idx.build_initial([tmp_files["sim_a"]])
        idx.add_file(tmp_files["new_1"])
        info = idx.info()
        assert info["files_in_pending_buffer"] == 1
        assert info["files_in_main_index"] == 1

    def test_add_file_pending_included_in_query(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=10)
        idx.build_initial([tmp_files["sim_a"]])
        idx.add_file(tmp_files["sim_b"])
        result = idx.query(tmp_files["main"])
        # Should return results from both main index AND pending buffer
        assert len(result.compare) == 2

    def test_auto_merge_on_threshold(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=2)
        idx.build_initial([tmp_files["sim_a"]])
        idx.add_file(tmp_files["new_1"])   # pending count = 1
        merged = idx.add_file(tmp_files["new_2"])   # pending count = 2 → triggers merge
        assert merged is True
        info = idx.info()
        assert info["files_in_pending_buffer"] == 0
        assert info["files_in_main_index"] == 3   # sim_a + new_1 + new_2

    def test_force_merge(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=100)
        idx.build_initial([tmp_files["sim_a"]])
        idx.add_file(tmp_files["new_1"])
        idx.force_merge()
        info = idx.info()
        assert info["files_in_pending_buffer"] == 0
        assert info["files_in_main_index"] == 2

    def test_remove_from_pending(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=100)
        idx.build_initial([tmp_files["sim_a"]])
        idx.add_file(tmp_files["new_1"])
        found = idx.remove_file(tmp_files["new_1"])
        assert found is True
        assert idx.info()["files_in_pending_buffer"] == 0

    def test_remove_from_main_index(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=100)
        idx.build_initial([tmp_files["sim_a"], tmp_files["sim_b"]])
        found = idx.remove_file(tmp_files["sim_a"])
        assert found is True
        assert idx.info()["files_in_main_index"] == 1

    def test_remove_nonexistent_returns_false(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=100)
        idx.build_initial([tmp_files["sim_a"]])
        found = idx.remove_file(tmp_files["unrela"])
        assert found is False

    def test_save_and_load(self, tmp_files, tmp_path):
        idx = DynamicFileIndex(merge_threshold=10)
        idx.build_initial([tmp_files["sim_a"], tmp_files["sim_b"]])
        idx.add_file(tmp_files["new_1"])   # in pending

        save_path = tmp_path / "test.dyn"
        idx.save(save_path)

        loaded = DynamicFileIndex.load(save_path)
        info = loaded.info()
        assert info["files_in_main_index"] == 2
        assert info["files_in_pending_buffer"] == 1

    def test_save_load_query_consistent(self, tmp_files, tmp_path):
        idx = DynamicFileIndex(merge_threshold=10)
        idx.build_initial([tmp_files["sim_a"], tmp_files["sim_b"]])
        r1 = idx.query(tmp_files["main"])

        idx.save(tmp_path / "idx.dyn")
        idx2 = DynamicFileIndex.load(tmp_path / "idx.dyn")
        r2 = idx2.query(tmp_files["main"])

        assert len(r1.compare) == len(r2.compare)

    def test_query_text(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=10)
        idx.build_initial([tmp_files["sim_a"], tmp_files["sim_b"]])
        result = idx.query_text("Python data science", top_n=2)
        assert len(result.compare) <= 2

    def test_query_with_threshold(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=10)
        idx.build_initial([tmp_files["sim_a"], tmp_files["unrela"]])
        result = idx.query(tmp_files["main"], threshold=5.0)
        for m in result.compare:
            assert m.percentage >= 5.0

    def test_empty_raises(self, tmp_files):
        idx = DynamicFileIndex()
        with pytest.raises(RuntimeError):
            idx.query(tmp_files["main"])

    def test_add_files_bulk(self, tmp_files):
        idx = DynamicFileIndex(merge_threshold=100)
        idx.build_initial([tmp_files["sim_a"]])
        idx.add_files([tmp_files["new_1"], tmp_files["new_2"]])
        assert idx.info()["files_in_pending_buffer"] == 2

    def test_with_cache(self, tmp_files, tmp_cache):
        idx = DynamicFileIndex(merge_threshold=10, cache=tmp_cache)
        idx.build_initial([tmp_files["sim_a"], tmp_files["sim_b"]])
        r1 = idx.query(tmp_files["main"])
        r2 = idx.query(tmp_files["main"])   # second call uses cache
        assert len(r1.compare) == len(r2.compare)


# ---------------------------------------------------------------------------
# IndexRegistry
# ---------------------------------------------------------------------------

class TestIndexRegistry:

    def test_create_and_list(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("legal")
        assert "legal" in reg.list_groups()

    def test_create_duplicate_raises(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("legal")
        with pytest.raises(ValueError):
            reg.create_group("legal")

    def test_create_overwrite(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("legal")
        reg.create_group("legal", overwrite=True)  # should not raise
        assert "legal" in reg.list_groups()

    def test_create_with_initial_files(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("docs", initial_files=[tmp_files["sim_a"], tmp_files["sim_b"]])
        info = reg.group_info("docs")
        assert info["files_in_main_index"] == 2

    def test_add_file_to_group(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("eng", initial_files=[tmp_files["sim_a"]])
        reg.add_file("eng", tmp_files["new_1"])
        info = reg.group_info("eng")
        assert info["files_in_pending_buffer"] == 1

    def test_add_files_bulk(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("eng", initial_files=[tmp_files["sim_a"]])
        reg.add_files("eng", [tmp_files["new_1"], tmp_files["new_2"]])
        info = reg.group_info("eng")
        assert info["files_in_pending_buffer"] == 2

    def test_remove_file(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("docs", initial_files=[tmp_files["sim_a"], tmp_files["sim_b"]])
        found = reg.remove_file("docs", tmp_files["sim_a"])
        assert found is True
        assert reg.group_info("docs")["files_in_main_index"] == 1

    def test_query_single_group(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("docs", initial_files=[tmp_files["sim_a"], tmp_files["sim_b"]])
        result = reg.query("docs", tmp_files["main"])
        assert len(result.compare) == 2

    def test_query_all_groups(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("legal", initial_files=[tmp_files["sim_a"]])
        reg.create_group("engineering", initial_files=[tmp_files["sim_b"]])
        results = reg.query_all(tmp_files["main"])
        assert "legal" in results
        assert "engineering" in results
        assert len(results["legal"].compare) == 1
        assert len(results["engineering"].compare) == 1

    def test_delete_group(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("temp")
        reg.delete_group("temp")
        assert "temp" not in reg.list_groups()

    def test_delete_removes_file(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("temp")
        idx_file = tmp_path / "registry" / "groups" / "temp.dyn"
        assert idx_file.exists()
        reg.delete_group("temp")
        assert not idx_file.exists()

    def test_group_not_found_raises(self, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        with pytest.raises(GroupNotFoundError):
            reg.query("nonexistent", "some_file.txt")

    def test_persists_across_reload(self, tmp_files, tmp_path):
        reg_dir = tmp_path / "registry"
        reg = IndexRegistry(reg_dir)
        reg.create_group("docs", initial_files=[tmp_files["sim_a"]])
        reg.add_file("docs", tmp_files["new_1"])

        # Reload from same directory
        reg2 = IndexRegistry(reg_dir)
        assert "docs" in reg2.list_groups()
        info = reg2.group_info("docs")
        assert info["total_files"] == 2

    def test_force_merge_via_registry(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry", merge_threshold=100)
        reg.create_group("big", initial_files=[tmp_files["sim_a"]])
        reg.add_file("big", tmp_files["new_1"])
        reg.force_merge("big")
        info = reg.group_info("big")
        assert info["files_in_pending_buffer"] == 0
        assert info["files_in_main_index"] == 2

    def test_cache_stats(self, tmp_path):
        reg = IndexRegistry(tmp_path / "registry", shared_cache=True)
        stats = reg.cache_stats()
        assert "entries" in stats

    def test_query_text(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("docs", initial_files=[tmp_files["sim_a"], tmp_files["sim_b"]])
        result = reg.query_text("docs", "Python data science", top_n=2)
        assert len(result.compare) <= 2

    def test_all_info(self, tmp_files, tmp_path):
        reg = IndexRegistry(tmp_path / "registry")
        reg.create_group("a", initial_files=[tmp_files["sim_a"]])
        reg.create_group("b", initial_files=[tmp_files["sim_b"]])
        all_info = reg.all_info()
        assert "a" in all_info
        assert "b" in all_info
