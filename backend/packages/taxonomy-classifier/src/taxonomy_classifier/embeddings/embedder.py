"""Dense embeddings (OpenAI embeddings API by default) with the canonical cache.

Every UNIQUE uncached text is sent to the API once; duplicates in the same request reuse the vector that was stored
for that text (see cache.py). Vectors are returned L2-normalised.

`transport` can be injected (tests, other providers): a callable list[str] -> (list[np.ndarray], total_tokens).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable

import numpy as np

from ..llm.client import api_key
from .cache import EmbeddingCache, norm

Transport = Callable[[list[str]], tuple[list[np.ndarray], int]]


class Embedder:
    def __init__(self, cfg: dict, ledger=None, transport: Transport | None = None):
        e = cfg["embedding"]
        self.model, self.dim, self.batch, self.price = e["model"], e["dimensions"], e["batch_size"], e.get("price_per_m_tokens_usd", 0.0)
        self.cache = EmbeddingCache(e["cache_path"], self.model, self.dim)
        self.ledger = ledger
        self.transport = transport or self._openai
        self.stats = {"requested": 0, "cache_hits": 0, "api_embedded": 0, "api_tokens": 0, "api_requests": 0, "duplicate_texts_reused": 0}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        out: list[np.ndarray | None] = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            self.stats["requested"] += 1
            v = self.cache.get(t)
            if v is not None:
                out[i] = v; self.stats["cache_hits"] += 1
            else:
                todo.append(i)
        first: dict[str, int] = {}
        for i in todo:                                   # embed each UNIQUE text once (API is not bit-deterministic)
            first.setdefault(self.cache.key(texts[i]), i)
        uniq = list(first.values())
        self.stats["duplicate_texts_reused"] += len(todo) - len(uniq)
        for s in range(0, len(uniq), self.batch):
            idx = uniq[s:s + self.batch]
            batch = [norm(texts[i]) for i in idx]
            if self.ledger is not None:
                worst = sum(len(t) for t in batch) / 3 * self.price / 1e6
                if not self.ledger.can_spend(worst):
                    raise RuntimeError("embedding call refused: run cost cap would be exceeded")
            t0 = time.perf_counter()
            vecs, toks = self.transport(batch)
            if self.ledger is not None:
                self.ledger.add("embedding", toks * self.price / 1e6, {"prompt_tokens": toks}, time.perf_counter() - t0)
            self.stats["api_embedded"] += len(batch); self.stats["api_tokens"] += toks; self.stats["api_requests"] += 1
            for i, v in zip(idx, vecs):
                self.cache.put_if_absent(texts[i], v)
            self.cache.commit()
        for i in todo:                                   # every uncached position (incl. duplicates) reads the canonical vector
            out[i] = self.cache.get(texts[i])
        X = np.stack(out).astype(np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def _openai(self, batch: list[str]):
        body = json.dumps({"model": self.model, "input": batch, "dimensions": self.dim}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/embeddings", data=body,
                                     headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
        for i in range(5):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=300)); break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and i < 4:
                    time.sleep(4 * 2 ** i); continue
                raise
        data = sorted(r["data"], key=lambda d: d["index"])
        return [np.array(d["embedding"], dtype=np.float32) for d in data], r["usage"]["total_tokens"]
