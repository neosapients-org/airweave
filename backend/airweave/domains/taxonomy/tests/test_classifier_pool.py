"""Dedicated classifier threads (plan A7 classifier_pool.py; design review finding 4).

The stand-in classifier behaves like the package's SQLite cache: it refuses to be used from any
thread but the one that created it. The real package and cache are exercised in
``test_classifier_integration.py``.
"""

import asyncio
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.classifier_pool import ClassifierPool
from airweave.domains.taxonomy.mapping import RegistryMap


class WrongThread(Exception):
    pass


class StandInClassifier:
    created: list = []

    def __init__(self, taxonomy_path, config, transport=None):
        self.thread = threading.get_ident()
        self.config = config
        self.taxonomy = types.SimpleNamespace(path=taxonomy_path)
        self.embedder = types.SimpleNamespace(cache=types.SimpleNamespace(close=self._close))
        self.closed = False
        self.delay = float(config.get("_delay", 0))
        StandInClassifier.created.append(self)

    def _close(self):
        self.closed = True

    def classify(self, rows):
        if threading.get_ident() != self.thread:
            raise WrongThread("used from another thread")
        if self.delay:
            time.sleep(self.delay)
        return types.SimpleNamespace(
            enriched=[{"chunk_id": r["chunk_id"], "thread": self.thread} for r in rows])


@pytest.fixture
def stand_in(monkeypatch):
    StandInClassifier.created = []
    integ = types.ModuleType("taxonomy_classifier.integration.airweave")
    integ.TaxonomyClassifier = StandInClassifier
    contract = types.ModuleType("taxonomy_classifier.contract")
    contract.to_contract = lambda e, h: dict(e)
    for name, mod in (("taxonomy_classifier", types.ModuleType("taxonomy_classifier")),
                      ("taxonomy_classifier.integration", types.ModuleType("x")),
                      ("taxonomy_classifier.integration.airweave", integ),
                      ("taxonomy_classifier.contract", contract)):
        monkeypatch.setitem(sys.modules, name, mod)
    return StandInClassifier


def _taxonomy(version="acme-2026.10", sha="a" * 64, **config) -> ResolvedTaxonomy:
    return ResolvedTaxonomy("neo-acme", "t-acme", version, sha, 1, "calibrated", None,
                            Path("/tmp/none"), {"embedding": {}, **config},
                            RegistryMap(version, {}, {}))


def _rows(n, prefix="c"):
    return [{"chunk_id": f"{prefix}{i}", "document_id": "d", "text": "Acme"} for i in range(n)]


class TestThreads:
    async def test_each_classifier_only_used_on_its_thread(self, tmp_path, stand_in):
        pool = ClassifierPool(2, tmp_path)
        results = await asyncio.gather(*[pool.classify(_taxonomy(), _rows(3, f"b{i}_"), 10)
                                         for i in range(12)])
        assert all("thread" in r for batch in results for r in batch)
        assert len(stand_in.created) == 2  # one classifier per thread for the version
        assert len({c.thread for c in stand_in.created}) == 2
        pool.close()

    async def test_round_robin(self, tmp_path, stand_in):
        pool = ClassifierPool(3, tmp_path)
        threads = [(await pool.classify(_taxonomy(), _rows(1), 10))[0]["thread"]
                   for _ in range(6)]
        assert threads[:3] == threads[3:] and len(set(threads)) == 3
        pool.close()

    async def test_cache_file_per_process_and_thread(self, tmp_path, stand_in):
        pool = ClassifierPool(2, tmp_path / "cache")
        for _ in range(2):
            await pool.classify(_taxonomy(), _rows(1), 10)
        paths = sorted(c.config["embedding"]["cache_path"] for c in stand_in.created)
        assert paths == [str(tmp_path / "cache" / f"{os.getpid()}-{i}.sqlite") for i in (0, 1)]
        pool.close()

    async def test_versions_cached_with_lru(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path, classifiers_per_thread=2)
        for v in ("v1", "v2", "v1", "v3"):
            await pool.classify(_taxonomy(version=v), _rows(1), 10)
        assert [c.taxonomy.path for c in stand_in.created] == [Path("/tmp/none")] * 3
        v1, v2, v3 = stand_in.created
        assert v2.closed and not v1.closed and not v3.closed  # v2 was least recently used
        pool.close()

    async def test_same_version_new_checksum_is_a_new_classifier(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path)
        await pool.classify(_taxonomy(sha="a" * 64), _rows(1), 10)
        await pool.classify(_taxonomy(sha="b" * 64), _rows(1), 10)
        assert len(stand_in.created) == 2
        pool.close()


class TestTimeout:
    async def test_timeout_raises_and_next_batch_waits_behind(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path)
        slow = _taxonomy(_delay=0.6)
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await pool.classify(slow, _rows(1), 0.1)
        result = await pool.classify(slow, _rows(1, "next"), 10)
        assert result[0]["chunk_id"] == "next0"
        assert time.monotonic() - start >= 1.1  # the timed-out batch finished first
        pool.close()

    def test_needs_a_thread(self, tmp_path):
        with pytest.raises(ValueError):
            ClassifierPool(0, tmp_path)
