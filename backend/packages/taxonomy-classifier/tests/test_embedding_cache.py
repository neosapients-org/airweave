"""Canonical-vector rule: identical text is embedded once and always maps to the same cached vector."""

import numpy as np

from taxonomy_classifier.embeddings.cache import EmbeddingCache
from taxonomy_classifier.embeddings.embedder import Embedder


class NonDeterministicAPI:
    """Mimics the real API: the same text returns a slightly different vector on every request."""

    def __init__(self):
        self.calls, self.sent, self.rng = 0, [], np.random.default_rng(0)

    def __call__(self, batch):
        self.calls += 1; self.sent += batch
        return [np.ones(8, np.float32) + self.rng.normal(scale=1e-3, size=8).astype(np.float32) for _ in batch], 10


def mk(cfg, api):
    cfg = {**cfg, "embedding": {**cfg["embedding"], "dimensions": 8}}
    return Embedder(cfg, None, api)


def test_duplicate_text_embedded_once_same_vector(cfg):
    api = NonDeterministicAPI(); e = mk(cfg, api)
    X = e.embed(["same text", "other text", "same   text", "same text"])
    assert api.sent.count("same text") == 1                          # unique text sent once (whitespace-normalised)
    assert np.array_equal(X[0], X[2]) and np.array_equal(X[0], X[3])   # exactly the same vector
    assert e.stats["duplicate_texts_reused"] == 2


def test_cached_vector_is_canonical_across_calls(cfg):
    api = NonDeterministicAPI(); e = mk(cfg, api)
    a = e.embed(["alpha"])[0]
    b = e.embed(["alpha", "beta"])[0]
    assert api.sent.count("alpha") == 1 and np.array_equal(a, b)
    e2 = mk(cfg, api)                                                  # new process, same cache file
    assert np.array_equal(e2.embed(["alpha"])[0], a) and api.sent.count("alpha") == 1


def test_first_write_wins(tmp_path):
    c = EmbeddingCache(tmp_path / "c.sqlite", "m", 3)
    c.put_if_absent("x", np.array([1, 0, 0], np.float32)); c.put_if_absent("x", np.array([0, 1, 0], np.float32)); c.commit()
    assert c.get("x").tolist() == [1, 0, 0]
