"""
filecompare/registry.py
-----------------------
IndexRegistry: manage multiple named DynamicFileIndex groups.

Each "group" is an independent DynamicFileIndex stored as its own file.
The registry itself is a thin JSON manifest that tracks which groups exist.

Why a registry?
---------------
Your groups are dynamic — new groups are created, files are uploaded into
specific groups, and you want to query across one or all groups.

Directory layout managed by registry:

    /your/index/dir/
    ├── registry.json          ← manifest (group names + metadata)
    ├── groups/
    │   ├── legal.dyn          ← DynamicFileIndex for "legal" group
    │   ├── engineering.dyn
    │   └── finance.dyn
    └── cache.db               ← shared VectorCache (optional)

Usage
-----
from filecompare import IndexRegistry

# Open (or create) a registry
reg = IndexRegistry("/data/my_indexes")

# Create a group
reg.create_group("legal")

# Add files to a group (e.g., on every upload)
reg.add_file("legal", "/uploads/contract_v1.pdf.txt")
reg.add_file("legal", "/uploads/contract_v2.pdf.txt")

# Query within a group
result = reg.query("legal", "/uploads/new_contract.txt", top_n=10)

# Query across ALL groups
results = reg.query_all("/uploads/mystery_doc.txt", top_n=5)

# List groups
print(reg.list_groups())

# Remove a file from a group
reg.remove_file("legal", "/uploads/old_file.txt")

# Delete a group entirely
reg.delete_group("legal")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .cache import VectorCache
from .index import DynamicFileIndex
from .types import FileMatch, MultiResult


class GroupNotFoundError(KeyError):
    pass


class IndexRegistry:
    """
    Manages a collection of named DynamicFileIndex groups.

    Parameters
    ----------
    base_dir        : Root directory where all indexes and the manifest are stored.
    merge_threshold : Default merge threshold for newly created groups.
    shared_cache    : If True, all groups share a single VectorCache (recommended).
    """

    MANIFEST_FILE = "registry.json"
    GROUPS_DIR = "groups"
    CACHE_FILE = "cache.db"

    def __init__(
        self,
        base_dir: str | Path,
        merge_threshold: int = 50,
        shared_cache: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / self.GROUPS_DIR).mkdir(exist_ok=True)

        self.default_merge_threshold = merge_threshold

        # Shared cache across all groups — files in multiple groups only processed once
        self._cache: Optional[VectorCache] = None
        if shared_cache:
            self._cache = VectorCache(self.base_dir / self.CACHE_FILE)

        # In-memory loaded indexes (lazy-loaded on first access)
        self._loaded: Dict[str, DynamicFileIndex] = {}

        # Load or create manifest
        self._manifest: dict = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    @property
    def _manifest_path(self) -> Path:
        return self.base_dir / self.MANIFEST_FILE

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {"groups": {}}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps(self._manifest, indent=2)
        )

    def _group_path(self, name: str) -> Path:
        return self.base_dir / self.GROUPS_DIR / f"{name}.dyn"

    def _assert_exists(self, name: str) -> None:
        if name not in self._manifest["groups"]:
            raise GroupNotFoundError(
                f"Group '{name}' not found. "
                f"Available: {list(self._manifest['groups'].keys())}"
            )

    # ------------------------------------------------------------------
    # Group lifecycle
    # ------------------------------------------------------------------

    def create_group(
        self,
        name: str,
        initial_files: Optional[List[str | Path]] = None,
        merge_threshold: Optional[int] = None,
        overwrite: bool = False,
    ) -> DynamicFileIndex:
        """
        Create a new named group.

        Parameters
        ----------
        name            : Unique group name (e.g. "legal", "engineering").
        initial_files   : Optional seed files. Useful when migrating an existing corpus.
        merge_threshold : Override default merge_threshold for this group.
        overwrite       : If True, replace an existing group of the same name.

        Returns the created DynamicFileIndex.
        """
        if name in self._manifest["groups"] and not overwrite:
            raise ValueError(
                f"Group '{name}' already exists. "
                "Pass overwrite=True to replace it."
            )

        threshold = merge_threshold or self.default_merge_threshold
        idx = DynamicFileIndex(
            merge_threshold=threshold,
            cache=self._cache,
        )

        if initial_files:
            idx.build_initial(initial_files, verbose=False)

        idx.save(self._group_path(name))
        self._loaded[name] = idx

        self._manifest["groups"][name] = {
            "created_at": time.time(),
            "merge_threshold": threshold,
            "path": str(self._group_path(name)),
        }
        self._save_manifest()

        return idx

    def delete_group(self, name: str) -> None:
        """Delete a group and its index file."""
        self._assert_exists(name)
        path = self._group_path(name)
        if path.exists():
            path.unlink()
        self._loaded.pop(name, None)
        del self._manifest["groups"][name]
        self._save_manifest()

    def list_groups(self) -> List[str]:
        """Return names of all registered groups."""
        return list(self._manifest["groups"].keys())

    def group_info(self, name: str) -> dict:
        """Return metadata + DynamicFileIndex.info() for a group."""
        self._assert_exists(name)
        idx = self._get(name)
        meta = self._manifest["groups"][name].copy()
        meta.update(idx.info())
        return meta

    def all_info(self) -> dict:
        """Return info for every group."""
        return {name: self.group_info(name) for name in self.list_groups()}

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _get(self, name: str) -> DynamicFileIndex:
        """Load a group index into memory (lazy, cached)."""
        self._assert_exists(name)
        if name not in self._loaded:
            self._loaded[name] = DynamicFileIndex.load(
                self._group_path(name),
                cache=self._cache,
            )
        return self._loaded[name]

    def _save_group(self, name: str) -> None:
        """Persist a loaded group back to disk."""
        if name in self._loaded:
            self._loaded[name].save(self._group_path(name))

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def add_file(self, group: str, path: str | Path) -> bool:
        """
        Add a single file to a group.

        The file is staged in the pending buffer immediately and auto-saved.
        If the buffer hits merge_threshold, a full rebuild is triggered.

        Returns True if a merge was triggered.
        """
        idx = self._get(group)
        merged = idx.add_file(path)
        self._save_group(group)
        return merged

    def add_files(self, group: str, paths: List[str | Path]) -> bool:
        """
        Add multiple files to a group at once.
        More efficient than calling add_file() in a loop — only one
        merge check and one disk write regardless of how many files.

        Returns True if a merge was triggered.
        """
        idx = self._get(group)
        merged = idx.add_files(paths)
        self._save_group(group)
        return merged

    def remove_file(self, group: str, path: str | Path) -> bool:
        """
        Remove a file from a group.
        Returns True if the file was found and removed.
        """
        idx = self._get(group)
        found = idx.remove_file(path)
        if found:
            self._save_group(group)
        return found

    def force_merge(self, group: str) -> None:
        """Force an immediate rebuild of a group's main index from all pending files."""
        idx = self._get(group)
        idx.force_merge()
        self._save_group(group)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        group: str,
        main_file: str | Path,
        top_n: Optional[int] = None,
        threshold: float = 0.0,
    ) -> MultiResult:
        """
        Compare main_file against all files in a single group.

        Parameters
        ----------
        group     : Group name to search.
        main_file : File to compare.
        top_n     : Return only top-n matches.
        threshold : Exclude matches below this percentage.
        """
        return self._get(group).query(
            main_file,
            top_n=top_n,
            threshold=threshold,
        )

    def query_all(
        self,
        main_file: str | Path,
        top_n: Optional[int] = None,
        threshold: float = 0.0,
    ) -> Dict[str, MultiResult]:
        """
        Compare main_file against ALL groups simultaneously.

        Returns a dict mapping group_name → MultiResult.

        Useful for finding which group (and which files within it) most
        closely match the query document.

        Example
        -------
        results = reg.query_all("mystery_doc.txt", top_n=5, threshold=5.0)
        for group, result in results.items():
            print(f"--- {group} ---")
            for m in result.compare:
                print(f"  {m.percentage}%  {m.file}")
        """
        return {
            name: self.query(name, main_file, top_n=top_n, threshold=threshold)
            for name in self.list_groups()
        }

    def query_text(
        self,
        group: str,
        text: str,
        top_n: Optional[int] = None,
        threshold: float = 0.0,
    ) -> MultiResult:
        """Compare raw text against all files in a group (no file needed)."""
        return self._get(group).query_text(text, top_n=top_n, threshold=threshold)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        if self._cache:
            return self._cache.stats()
        return {"message": "No shared cache configured."}

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()
