"""
filemetric_engine
-----------
A Python package for comparing text files by similarity percentage.

Quick start
-----------

# --- Simple pair comparison ---
from filemetric_engine import compare_files
result = compare_files("essay.txt", "reference.txt")
print(result.to_dict())

# --- One vs many (small sets) ---
from filemetric_engine import compare_one_to_many
result = compare_one_to_many("submission.txt", ["ref1.txt", "ref2.txt", "ref3.txt"])
for m in result.compare:
    print(m.file, m.percentage)

# --- Scale: 1 query vs 10,000 files (use FileIndex) ---
from filemetric_engine import FileIndex, VectorCache

cache = VectorCache()                          # persists to ~/.filemetric_engine/cache.db
idx = FileIndex.build(base_files, cache=cache) # build once
idx.save("my_index.pkl")                       # save for later

idx = FileIndex.load("my_index.pkl")           # reload instantly
result = idx.query("new_file.txt", top_n=20)   # query in milliseconds
print(result.to_dict())
"""

from .cache import VectorCache
from .compare import compare_files, compare_one_to_many
from .index import DynamicFileIndex, FileIndex
from .registry import GroupNotFoundError, IndexRegistry
from .types import FileMatch, MultiResult, PairResult

__all__ = [
    # Core functions
    "compare_files",
    "compare_one_to_many",
    # Index (immutable, for bulk use)
    "FileIndex",
    # Dynamic index (incremental adds, for live systems)
    "DynamicFileIndex",
    # Registry (named groups, multi-tenant)
    "IndexRegistry",
    "GroupNotFoundError",
    # Cache
    "VectorCache",
    # Return types
    "PairResult",
    "MultiResult",
    "FileMatch",
]

__version__ = "1.0.0"
__author__ = "Gennifortune Technology Solutions Ltd"
