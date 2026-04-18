# filecompare

Compare text files by similarity percentage — with caching and large-scale indexing.

## Install

```bash
pip install scikit-learn numpy scipy
# Optional: for meaning-based (semantic) similarity
pip install sentence-transformers
```

Or install as a local package from this folder:
```bash
pip install -e .
```

---

## Usage

### 1. Simple pair comparison

```python
from filecompare import compare_files

result = compare_files("essay.txt", "reference.txt")
print(result.to_dict())
# {"file_1": "essay.txt", "file_2": "reference.txt", "common_in_percentage": 67.34}
```

### 2. One vs many (small sets, < 500 files)

```python
from filecompare import compare_one_to_many

result = compare_one_to_many(
    "submission.txt",
    ["ref1.txt", "ref2.txt", "ref3.txt"],
    top_n=10,         # return only top 10
    threshold=5.0,    # exclude < 5% matches`
)

for m in result.compare:
    print(f"{m.file}  →  {m.percentage}%")

print(result.to_dict())
```

### 3. Large scale: 1 query vs 10,000 files (use FileIndex)

```python
from filecompare import FileIndex, VectorCache

# Step 1 — create a cache (persists to ~/.filecompare/cache.db)
cache = VectorCache()

# Step 2 — build index once (reads & vectorises all files; uses cache)
import glob
base_files = glob.glob("/data/corpus/**/*.txt", recursive=True)

idx = FileIndex.build(base_files, cache=cache, verbose=True)
idx.save("corpus_index.pkl")   # save to disk

# Step 3 — reload instantly on next run
idx = FileIndex.load("corpus_index.pkl")

# Step 4 — query in milliseconds
result = idx.query("new_document.txt", top_n=20, threshold=5.0)
for m in result.compare:
    print(f"{m.file}  {m.percentage}%")
```

### 4. Add cache to any function

```python
from filecompare import compare_files, compare_one_to_many, VectorCache

cache = VectorCache()  # reuse across calls

# File reads are deduplicated — same file content is only processed once
r1 = compare_files("a.txt", "b.txt", cache=cache)
r2 = compare_files("a.txt", "c.txt", cache=cache)  # a.txt served from cache
```

### 5. Query raw text (no file needed)

```python
idx = FileIndex.load("corpus_index.pkl")
result = idx.query_text("some raw text content here", top_n=5)
```

---

## Performance at scale

| Files | Approach | Build time | Query time |
|-------|----------|-----------|------------|
| < 100 | `compare_one_to_many` | — | < 100ms |
| 100–1000 | `compare_one_to_many` | — | 100ms–2s |
| 1000–50000 | `FileIndex.build` + `.query` | 30s–5min* | **< 50ms** |

\* With cache, rebuild is near-instant for unchanged files.

### Cache behaviour
- Cache is keyed on **SHA-256 of file content** — not the filename.
- If a file changes, it's automatically re-processed on next build.
- Same content at different paths → processed once.
- Cache database lives at `~/.filecompare/cache.db` by default.

```python
cache = VectorCache()
print(cache.stats())
# {"entries": 9832, "text_bytes": 48291200, "vector_bytes": 0, "db_path": "..."}

cache.clear()        # wipe all
cache.invalidate(h)  # remove one entry by hash
```

---

## API reference

### `compare_files(file_1, file_2, cache=None) → PairResult`
### `compare_one_to_many(main_file, base_files, cache=None, top_n=None, threshold=0.0, sort=True) → MultiResult`
### `FileIndex.build(base_files, cache=None, max_workers=8, verbose=False) → FileIndex`
### `FileIndex.load(path) → FileIndex`
### `FileIndex.save(path)`
### `FileIndex.query(main_file, cache=None, top_n=None, threshold=0.0, sort=True) → MultiResult`
### `FileIndex.query_text(text, top_n=None, threshold=0.0) → MultiResult`
### `FileIndex.add_files(new_files, cache=None) → FileIndex`
### `FileIndex.info() → dict`

---

## Run tests

```bash
pip install pytest
pytest tests/ -v
```
