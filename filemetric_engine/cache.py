"""
filemetric_engine/cache.py
--------------------
Persistent, hash-based cache for processed file vectors/text.

How it works:
- Each file is identified by SHA-256 of its raw bytes (content-addressed).
- Cached entries are stored in a SQLite database (single file, zero config).
- Cache entries store: file_hash → (cleaned_text, optional_embedding_bytes)
- If a file's content changes, its hash changes → cache miss → re-process.
- If the same file appears under 10 different paths, it's only processed once.

Cache DB schema:
    CREATE TABLE vectors (
        hash      TEXT PRIMARY KEY,
        text      TEXT NOT NULL,
        vector    BLOB,           -- pickled numpy array (TF-IDF not stored here,
                                  --   only semantic embeddings are stored per-file)
        created   REAL
    )
"""

from __future__ import annotations

import hashlib
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


_DEFAULT_CACHE_PATH = Path.home() / ".filemetric_engine" / "cache.db"


class VectorCache:
    """
    SQLite-backed content-addressable cache.

    Parameters
    ----------
    db_path : Where to store the SQLite file. Defaults to ~/.filemetric_engine/cache.db
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_CACHE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._get_conn()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS file_cache (
                hash      TEXT PRIMARY KEY,
                text      TEXT NOT NULL,
                vector    BLOB,
                created   REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hash ON file_cache (hash)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Return SHA-256 hex digest of raw bytes."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_file(path: str | Path) -> str:
        """Return SHA-256 hex digest of a file on disk."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()

    def get(self, file_hash: str) -> Optional[tuple[str, Optional[np.ndarray]]]:
        """
        Retrieve cached (cleaned_text, embedding_or_None) for a hash.
        Returns None on cache miss.
        """
        row = self._conn.execute(
            "SELECT text, vector FROM file_cache WHERE hash = ?",
            (file_hash,),
        ).fetchone()

        if row is None:
            return None

        text = row[0]
        vector = pickle.loads(row[1]) if row[1] else None
        return text, vector

    def set(
        self,
        file_hash: str,
        text: str,
        vector: Optional[np.ndarray] = None,
    ) -> None:
        """Store a (cleaned_text, optional_embedding) entry."""
        vector_blob = pickle.dumps(vector) if vector is not None else None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO file_cache (hash, text, vector, created)
            VALUES (?, ?, ?, ?)
            """,
            (file_hash, text, vector_blob, time.time()),
        )
        self._conn.commit()

    def has(self, file_hash: str) -> bool:
        """Check existence without loading data."""
        row = self._conn.execute(
            "SELECT 1 FROM file_cache WHERE hash = ?", (file_hash,)
        ).fetchone()
        return row is not None

    def invalidate(self, file_hash: str) -> None:
        """Remove a single entry."""
        self._conn.execute(
            "DELETE FROM file_cache WHERE hash = ?", (file_hash,)
        )
        self._conn.commit()

    def clear(self) -> None:
        """Wipe entire cache."""
        self._conn.execute("DELETE FROM file_cache")
        self._conn.commit()

    def stats(self) -> dict:
        """Return cache statistics."""
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(LENGTH(text)), SUM(LENGTH(COALESCE(vector,''))) "
            "FROM file_cache"
        ).fetchone()
        return {
            "entries": row[0] or 0,
            "text_bytes": row[1] or 0,
            "vector_bytes": row[2] or 0,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
