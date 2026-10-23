"""Content-addressed embedding cache (SQLite).

key = SHA256(model | dimensions | whitespace-normalised text)

CANONICAL-VECTOR RULE (do not regress): the embeddings API is not bit-deterministic, so the same text embedded twice
can come back with slightly different vectors. The cache stores exactly ONE vector per key (INSERT OR IGNORE: the first
vector written wins) and every lookup of that text returns that vector. Without this, duplicate texts inside a corpus
get different vectors, UMAP/HDBSCAN see a different input, and a rerun stops being reproducible.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def cache_key(model: str, dim: int, text: str) -> str:
    return hashlib.sha256(f"{model}|{dim}|{norm(text)}".encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path, model: str, dim: int, read_only: bool = False):
        self.model, self.dim, self.read_only = model, dim, read_only
        p = Path(path)
        if read_only:
            self.db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(p)
            self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, model TEXT, dim INT, tokens INT, vector BLOB)")

    def key(self, text: str) -> str:
        return cache_key(self.model, self.dim, text)

    def get(self, text: str) -> np.ndarray | None:
        key = self.key(text)
        row = self.db.execute("SELECT vector FROM emb WHERE key=? AND model=? AND dim=?", (key, self.model, self.dim)).fetchone()
        if not row:
            return None
        v = np.frombuffer(row[0], dtype=np.float32)
        if v.shape == (self.dim,) and np.isfinite(v).all() and np.any(v):
            return v
        # A damaged row (wrong length, NaN, all zeros) is a miss: it is removed so the fresh vector can take its place
        # (first write wins, so it would otherwise stay); a read-only cache just ignores it.
        if not self.read_only:
            self.db.execute("DELETE FROM emb WHERE key=?", (key,))
        return None

    def put_if_absent(self, text: str, vec: np.ndarray):
        """First write wins: an existing vector for this text is never replaced."""
        self.db.execute("INSERT OR IGNORE INTO emb VALUES (?,?,?,?,?)",
                        (self.key(text), self.model, self.dim, -1, np.asarray(vec, dtype=np.float32).tobytes()))

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()
