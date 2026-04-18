# filemetric-engine

> A Python library for comparing text files by similarity percentage — with SHA-256 content caching, persistent TF-IDF indexing, incremental updates, and named group management.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen)]() [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/0x0pharaoh/filemetric-engine/pulls)

---

## What is filemetric-engine?

**filemetric-engine** is a local, dependency-light Python library that measures how similar two or more text files are to each other, returning a percentage score you can use directly in your application.

It scales from a quick two-file comparison all the way up to querying a single document against a corpus of 10,000+ files in under 50ms.

Everything runs locally. No API keys, no external services, no data leaves your machine.

**Common use cases:**

- Plagiarism and duplicate detection
- Contract and legal document comparison
- Code similarity analysis
- Document deduplication pipelines
- Research paper matching

---

## Table of Contents

- [filemetric-engine](#filemetric-engine)
  - [What is filemetric-engine?](#what-is-filemetric-engine)
  - [Table of Contents](#table-of-contents)
  - [How It Works](#how-it-works)
    - [Similarity algorithm](#similarity-algorithm)
    - [Two-tier index design](#two-tier-index-design)
    - [Caching](#caching)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Usage Guide](#usage-guide)
    - [1. Simple Pair Comparison](#1-simple-pair-comparison)
    - [2. One vs Many](#2-one-vs-many)
    - [3. Large Scale with FileIndex](#3-large-scale-with-fileindex)
    - [4. Dynamic Index — Incremental Uploads](#4-dynamic-index--incremental-uploads)
    - [5. IndexRegistry — Named Groups](#5-indexregistry--named-groups)
    - [6. VectorCache](#6-vectorcache)
    - [7. Query Raw Text](#7-query-raw-text)
  - [API Reference](#api-reference)
    - [`compare_files`](#compare_files)
    - [`compare_one_to_many`](#compare_one_to_many)
    - [`FileIndex`](#fileindex)
    - [`DynamicFileIndex`](#dynamicfileindex)
    - [`IndexRegistry`](#indexregistry)
    - [Return types](#return-types)
  - [Performance Guide](#performance-guide)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [Testing](#testing)
  - [Author](#author)
  - [License](#license)

---

## How It Works

### Similarity algorithm

filemetric-engine uses **TF-IDF (Term Frequency-Inverse Document Frequency)** with unigram + bigram tokenisation, followed by **cosine similarity**. This is the standard approach for text overlap detection — fast, interpretable, and produces consistent results without requiring a GPU or cloud service.

A score of `0%` means the two documents share no common terms. A score of `100%` means the documents are identical.

### Two-tier index design

For dynamic corpora where new files are uploaded frequently, the engine uses a two-tier architecture to avoid expensive full rebuilds on every write:

```
New file uploaded
      |
      v
 pending buffer        <-- O(1) write, instant
      |
      | (when buffer reaches merge_threshold)
      v
 main FileIndex        <-- full TF-IDF matrix, rebuilt automatically
      |
      +-- queries check both layers and merge results
```

### Caching

Files are identified by SHA-256 hash of their raw bytes, not their filename. Processed text is stored in a local SQLite database. If a file has not changed since the last run, it is served from cache with no disk read or re-processing.

---

## Installation

**Requirements:** Python 3.9 or higher

```bash
# 1. Clone the repository
git clone https://github.com/0x0pharaoh/filemetric-engine.git
cd filemetric-engine

# 2. Create a virtual environment
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 3. Install
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `scikit-learn` | TF-IDF vectoriser and cosine similarity |
| `numpy` | Numerical operations |
| `scipy` | Sparse matrix storage (memory efficient at scale) |

**Optional — semantic / meaning-based similarity:**

```bash
pip install sentence-transformers
```

Semantic mode compares documents by meaning rather than word overlap — useful when documents paraphrase the same ideas in different words. Requires more memory and is slower than TF-IDF.

---

## Quick Start

```python
from filemetric_engine import compare_files

result = compare_files("document_a.txt", "document_b.txt")
print(result.to_dict())
# {
#   "file_1": "document_a.txt",
#   "file_2": "document_b.txt",
#   "common_in_percentage": 67.34
# }
```

---

## Usage Guide

### 1. Simple Pair Comparison

```python
from filemetric_engine import compare_files

result = compare_files("essay.txt", "reference.txt")

print(result.file_1)                # "essay.txt"
print(result.file_2)                # "reference.txt"
print(result.common_in_percentage)  # 67.34
print(result.to_dict())             # full dict
```

Pass a `VectorCache` to skip re-reading files you have already processed:

```python
from filemetric_engine import compare_files, VectorCache

cache = VectorCache()  # persists to ~/.filemetric_engine/cache.db

r1 = compare_files("a.txt", "b.txt", cache=cache)  # reads from disk
r2 = compare_files("a.txt", "c.txt", cache=cache)  # a.txt from cache
```

---

### 2. One vs Many

```python
from filemetric_engine import compare_one_to_many

result = compare_one_to_many(
    "submission.txt",
    ["ref1.txt", "ref2.txt", "ref3.txt", "ref4.txt"],
    top_n=3,        # return only top 3 (optional)
    threshold=5.0,  # exclude matches below 5% (optional)
    sort=True,      # sort highest to lowest (default)
)

for match in result.compare:
    print(f"{match.percentage}%  ->  {match.file}")

# 72.4%  ->  ref1.txt
# 31.1%  ->  ref3.txt
# 12.8%  ->  ref2.txt

import json
print(json.dumps(result.to_dict(), indent=2))
```

> For base file lists over ~500 files, use `FileIndex` instead for much faster repeated queries.

---

### 3. Large Scale with FileIndex

`FileIndex` vectorises all base files once, saves the TF-IDF matrix to disk, and lets you run queries in under 50ms regardless of corpus size.

```python
import glob
from filemetric_engine import FileIndex, VectorCache

cache = VectorCache()
base_files = glob.glob("/data/corpus/**/*.txt", recursive=True)

# Build once
idx = FileIndex.build(base_files, cache=cache, verbose=True)
# [FileIndex] Reading 10000 files (8 workers)...
# [FileIndex] Fitting TF-IDF vectorizer...
# [FileIndex] Done. 10000 docs x 198432 features

idx.save("corpus.pkl")
```

```python
# All subsequent runs: load instantly
idx = FileIndex.load("corpus.pkl")

result = idx.query("new_document.txt", top_n=20, threshold=5.0)

for match in result.compare:
    print(f"{match.percentage}%  {match.file}")
```

```python
print(idx.info())
# {
#   "files_indexed": 10000,
#   "vocab_size": 198432,
#   "matrix_shape": [10000, 198432],
#   "built_at": "2024-10-18T09:30:00"
# }
```

---

### 4. Dynamic Index — Incremental Uploads

`DynamicFileIndex` wraps `FileIndex` with a pending buffer so new files can be added without rebuilding the entire index on every upload.

**One-time setup:**

```python
from filemetric_engine import DynamicFileIndex, VectorCache

cache = VectorCache()

idx = DynamicFileIndex(
    merge_threshold=50,  # rebuild main index after 50 buffered files
    cache=cache,
)
idx.build_initial(existing_files)
idx.save("my_index.dyn")
```

**On every new file upload:**

```python
idx = DynamicFileIndex.load("my_index.dyn", cache=cache)
idx.add_file("/uploads/new_doc.txt")  # instant O(1) write
idx.save("my_index.dyn")              # auto-merges when threshold is hit
```

**Query:**

```python
result = idx.query("compare_this.txt", top_n=10, threshold=5.0)
```

**Add multiple files at once:**

```python
idx.add_files(["/uploads/doc_a.txt", "/uploads/doc_b.txt"])
idx.save("my_index.dyn")
```

**Force an immediate rebuild** (useful after a bulk import):

```python
idx.force_merge()
idx.save("my_index.dyn")
```

**Remove a file:**

```python
idx.remove_file("/uploads/old_doc.txt")
idx.save("my_index.dyn")
```

**Inspect state:**

```python
print(idx.info())
# {
#   "files_in_main_index": 950,
#   "files_in_pending_buffer": 12,
#   "total_files": 962,
#   "merge_threshold": 50,
#   "main_index_built_at": "2024-10-18T09:30:00",
#   "pending_files": ["/uploads/doc_x.txt", ...]
# }
```

**Choosing a merge threshold:**

| Upload rate | Recommended `merge_threshold` |
|---|---|
| A few files/day | `10 - 25` |
| Dozens per day | `50` (default) |
| Hundreds per day | `100 - 200` |
| Bulk batch import | Call `force_merge()` manually after the batch |

---

### 5. IndexRegistry — Named Groups

`IndexRegistry` manages multiple independent `DynamicFileIndex` instances as named groups, all sharing a single `VectorCache`.

Use this when your application handles multiple document categories, projects, or users.

**Directory layout (managed automatically):**

```
/your/index/dir/
├── registry.json        <- group manifest
├── groups/
│   ├── contracts.dyn
│   ├── invoices.dyn
│   └── legal.dyn
└── cache.db             <- shared VectorCache
```

```python
from filemetric_engine import IndexRegistry

with IndexRegistry("/data/my_indexes", merge_threshold=50) as reg:

    # Create groups
    reg.create_group("contracts", initial_files=glob.glob("/docs/contracts/*.txt"))
    reg.create_group("invoices",  initial_files=glob.glob("/docs/invoices/*.txt"))

    print(reg.list_groups())  # ["contracts", "invoices"]

    # Add a file on upload
    reg.add_file("contracts", "/uploads/new_contract.txt")

    # Add multiple files
    reg.add_files("contracts", ["/uploads/a.txt", "/uploads/b.txt"])

    # Query a single group
    result = reg.query("contracts", "/uploads/mystery_doc.txt", top_n=10)

    # Query ALL groups simultaneously
    all_results = reg.query_all("/uploads/mystery_doc.txt", top_n=5)
    for group_name, group_result in all_results.items():
        if group_result.compare:
            top = group_result.compare[0]
            print(f"[{group_name}] {top.percentage}%  {top.file}")

    # Remove a file
    reg.remove_file("contracts", "/docs/contracts/expired.txt")

    # Force rebuild
    reg.force_merge("contracts")

    # Delete a group
    reg.delete_group("invoices")

    # Metadata
    print(reg.group_info("contracts"))
    print(reg.cache_stats())
```

Reload in a new process or after a server restart:

```python
with IndexRegistry("/data/my_indexes") as reg:
    result = reg.query("contracts", "new_doc.txt")
```

> **Windows users:** Always use `with IndexRegistry(...) as reg:` or call `reg.close()` explicitly before deleting the index directory. This closes the SQLite connection cleanly and prevents `PermissionError: [WinError 32]`.

---

### 6. VectorCache

`VectorCache` stores the processed text of each file in a local SQLite database, keyed by SHA-256 of the file's raw bytes.

```python
from filemetric_engine import VectorCache

# Default location: ~/.filemetric_engine/cache.db
cache = VectorCache()

# Custom path
cache = VectorCache("./project/cache.db")

# Inspect
print(cache.stats())
# {
#   "entries": 9832,
#   "text_bytes": 48291200,
#   "vector_bytes": 0,
#   "db_path": "/home/user/.filemetric_engine/cache.db"
# }

# Invalidate one entry by content hash
file_hash = VectorCache.hash_file("document.txt")
cache.invalidate(file_hash)

# Wipe all entries
cache.clear()

cache.close()
```

Use as a context manager:

```python
with VectorCache() as cache:
    result = compare_files("a.txt", "b.txt", cache=cache)
```

**Behaviour:**

- Files are identified by content hash, not path — renaming a file does not invalidate its cache entry
- Changing a file's content changes its hash, triggering automatic re-processing on next use
- Identical content at two different paths is stored and processed only once
- Backed by SQLite with WAL mode enabled for safe concurrent reads

---

### 7. Query Raw Text

Every query method has a `_text` variant that accepts a raw string instead of a file path. Useful when content comes from a database, API response, or in-memory buffer.

```python
# Against a FileIndex
idx = FileIndex.load("corpus.pkl")
result = idx.query_text("some raw document text here", top_n=5)

# Against a DynamicFileIndex
idx = DynamicFileIndex.load("my_index.dyn")
result = idx.query_text("raw text to compare", top_n=10, threshold=5.0)

# Against a registry group
with IndexRegistry("/data/my_indexes") as reg:
    result = reg.query_text("contracts", "raw contract text here", top_n=5)

for match in result.compare:
    print(f"{match.percentage}%  {match.file}")
```

---

## API Reference

### `compare_files`

```python
compare_files(
    file_1: str | Path,
    file_2: str | Path,
    cache: VectorCache | None = None,
) -> PairResult
```

### `compare_one_to_many`

```python
compare_one_to_many(
    main_file: str | Path,
    base_files: list[str | Path],
    cache: VectorCache | None = None,
    top_n: int | None = None,
    threshold: float = 0.0,
    sort: bool = True,
) -> MultiResult
```

### `FileIndex`

| Method | Description |
|---|---|
| `FileIndex.build(base_files, cache=None, max_workers=8, verbose=False)` | Build index from files |
| `FileIndex.load(path)` | Load a saved index from disk |
| `.save(path)` | Save index to disk |
| `.query(main_file, cache=None, top_n=None, threshold=0.0, sort=True)` | Query by file path |
| `.query_text(text, top_n=None, threshold=0.0)` | Query by raw string |
| `.add_files(new_files, cache=None)` | Return a new index with additional files |
| `.info()` | Return index metadata |

### `DynamicFileIndex`

| Method | Description |
|---|---|
| `DynamicFileIndex(merge_threshold=50, cache=None, max_workers=8)` | Create instance |
| `.build_initial(base_files, verbose=False)` | Seed from bulk files |
| `.add_file(path, force_merge=False) -> bool` | Add one file (buffered) |
| `.add_files(paths, force_merge=False) -> bool` | Add many files (one merge check) |
| `.remove_file(path) -> bool` | Remove a file |
| `.force_merge()` | Trigger immediate rebuild |
| `.save(path)` | Persist to disk |
| `DynamicFileIndex.load(path, cache=None)` | Load from disk |
| `.query(main_file, top_n=None, threshold=0.0, sort=True)` | Query both index layers |
| `.query_text(text, top_n=None, threshold=0.0)` | Query by raw string |
| `.info() -> dict` | State metadata |

### `IndexRegistry`

| Method | Description |
|---|---|
| `IndexRegistry(base_dir, merge_threshold=50, shared_cache=True)` | Open or create registry |
| `.create_group(name, initial_files=None, merge_threshold=None, overwrite=False)` | Create a group |
| `.delete_group(name)` | Delete a group |
| `.list_groups() -> list[str]` | All group names |
| `.group_info(name) -> dict` | Metadata for one group |
| `.all_info() -> dict` | Metadata for all groups |
| `.add_file(group, path) -> bool` | Add a file to a group |
| `.add_files(group, paths) -> bool` | Add multiple files to a group |
| `.remove_file(group, path) -> bool` | Remove a file from a group |
| `.force_merge(group)` | Force rebuild for a group |
| `.query(group, main_file, top_n=None, threshold=0.0)` | Query a single group |
| `.query_all(main_file, top_n=None, threshold=0.0) -> dict` | Query all groups |
| `.query_text(group, text, top_n=None, threshold=0.0)` | Query raw text |
| `.cache_stats() -> dict` | Cache statistics |
| `.close()` | Close SQLite connection |

### Return types

```python
@dataclass
class PairResult:
    file_1: str
    file_2: str
    common_in_percentage: float  # 0.0 to 100.0
    def to_dict(self) -> dict: ...

@dataclass
class FileMatch:
    file: str
    percentage: float
    def to_dict(self) -> dict: ...

@dataclass
class MultiResult:
    file: str
    compare: list[FileMatch]
    def to_dict(self) -> dict: ...
    def top(self, n: int) -> MultiResult: ...
```

---

## Performance Guide

| Corpus size | Recommended approach | First build | Subsequent queries |
|---|---|---|---|
| < 100 files | `compare_one_to_many` | N/A | < 100ms |
| 100 - 1,000 | `compare_one_to_many` + cache | N/A | < 1s |
| 1,000 - 50,000 | `FileIndex` or `DynamicFileIndex` | 30s - 5min* | **< 50ms** |
| 50,000+ | `FileIndex` with `max_features` tuning | varies | < 200ms |

*On first build. With cache populated, subsequent rebuilds skip file I/O and only re-fit the TF-IDF matrix.

**Reducing memory at scale:**

The TF-IDF matrix is stored as a sparse matrix. For very large corpora you can reduce memory usage by capping the vocabulary size:

```python
idx = FileIndex.build(
    base_files,
    vectorizer_kwargs={"max_features": 50_000},  # default is 200_000
)
```

---

## Project Structure

```
filemetric-engine/
├── filemetric_engine/
│   ├── __init__.py        <- public API exports
│   ├── types.py           <- PairResult, MultiResult, FileMatch
│   ├── cache.py           <- VectorCache (SQLite, thread-safe, SHA-256)
│   ├── compare.py         <- compare_files(), compare_one_to_many()
│   ├── index.py           <- FileIndex, DynamicFileIndex
│   └── registry.py        <- IndexRegistry
├── tests/
│   ├── test_filemetric.py <- unit tests: core functions, cache, FileIndex
│   └── test_dynamic.py    <- unit tests: DynamicFileIndex, IndexRegistry
├── smoke_test.py          <- end-to-end manual test
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Contributing

Contributions are welcome and appreciated.

```bash
# Fork the repo, then clone your fork
git clone https://github.com/your-username/filemetric-engine.git
cd filemetric-engine

# Create a feature branch
git checkout -b feature/your-feature-name

# Install in dev mode
pip install -e ".[dev]"

# Make your changes, then run the tests
pytest tests/ -v

# Push and open a pull request
git push origin feature/your-feature-name
```

**Guidelines:**

- New features should include tests in `tests/`
- Keep public API changes backward-compatible where possible
- Confirm `pytest tests/ -v` shows 63 passed before submitting
- Open an [issue](https://github.com/0x0pharaoh/filemetric-engine/issues) first if you are planning a large or breaking change

---

## Testing

**Run the full test suite:**

```bash
pytest tests/ -v
```

**Run the end-to-end smoke test:**

```bash
python smoke_test.py
```

The smoke test exercises all layers: pair comparison, one-to-many, cache hit/miss, dynamic index with auto-merge, registry group lifecycle, and persistence across simulated restarts.

Expected output:

```
filemetric-engine -- smoke test
------------------------------------------------------------
  Created 6 sample documents in sample_docs/

============================================================
  TEST 1: Simple pair comparison
============================================================
...
============================================================
  ALL TESTS PASSED
============================================================
```

---

## Author

**Pharaoh**
GitHub: [@0x0pharaoh](https://github.com/0x0pharaoh)

For bugs and feature requests, please open an [issue](https://github.com/0x0pharaoh/filemetric-engine/issues).
For questions or collaboration, reach out via GitHub.

---

## License

MIT License — see [LICENSE](./LICENSE) for full terms.

If you find this project useful, consider leaving a [star on GitHub](https://github.com/0x0pharaoh/filemetric-engine).