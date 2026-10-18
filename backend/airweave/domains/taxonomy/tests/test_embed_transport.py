"""Classifier embeddings through Airweave's own dense embedder (batch 3, item 9).

The classifier calls its transport on a dedicated thread, so the transport hands each batch to
the event loop that owns the embedder and waits there. The embedder's own timeouts, retries,
concurrency limit and error translation apply; the transport adds a deadline for the batch.
"""

import asyncio
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from airweave.domains.embedders.exceptions import EmbedderRateLimitError
from airweave.domains.embedders.types import DenseEmbedding
from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.classifier_pool import ClassifierPool
from airweave.domains.taxonomy.embed_transport import AirweaveEmbedTransport, matches
from airweave.domains.taxonomy.mapping import RegistryMap
from airweave.domains.taxonomy.metrics import TaxonomyChunksCollector, TaxonomyTally


class FakeDense:
    """Stands in for ``DenseEmbedderProtocol``; records the thread of the loop it runs on."""

    def __init__(self, model="text-embedding-3-small", dimensions=4, delay=0.0, error=None):
        self.model_name = model
        self.dimensions = dimensions
        self.delay = delay
        self.error = error
        self.calls: list = []
        self.loop_thread = None

    async def embed_many(self, texts):
        self.loop_thread = threading.get_ident()
        self.calls.append(list(texts))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return [DenseEmbedding(vector=[float(len(t)), 1.0, 0.0, 0.5]) for t in texts]


async def run_on_thread(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


async def bound_call(transport, batch, timeout=5):
    """Bind and call on one worker thread, as the pool does."""
    loop = asyncio.get_running_loop()

    def work():
        transport.bind(loop, timeout)
        return transport(batch)

    return await loop.run_in_executor(None, work)


class TestTransport:
    async def test_returns_vectors_and_tokens_in_the_packages_shape(self):
        dense = FakeDense()
        transport = AirweaveEmbedTransport(dense)
        vectors, tokens = await bound_call(transport, ["Acme invoice", "Globex"])
        assert [type(v) for v in vectors] == [np.ndarray, np.ndarray]
        assert all(v.dtype == np.float32 and v.shape == (4,) for v in vectors)
        assert vectors[0][0] == len("Acme invoice")
        assert tokens > 0
        assert dense.calls == [["Acme invoice", "Globex"]]

    async def test_embedder_runs_on_the_event_loop_not_the_classifier_thread(self):
        dense = FakeDense()
        transport = AirweaveEmbedTransport(dense)
        caller = []
        loop = asyncio.get_running_loop()

        def work():
            transport.bind(loop, 5)
            caller.append(threading.get_ident())
            return transport(["Acme"])

        await loop.run_in_executor(None, work)
        assert dense.loop_thread == threading.get_ident()
        assert dense.loop_thread != caller[0]

    async def test_the_loop_stays_free_while_a_batch_waits(self):
        dense = FakeDense(delay=0.3)
        transport = AirweaveEmbedTransport(dense)
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        await bound_call(transport, ["Acme"])
        task.cancel()
        assert ticks >= 5

    async def test_embedder_errors_reach_the_classifier_unchanged(self):
        error = EmbedderRateLimitError("slow down", provider="openai", retry_after=1.0)
        transport = AirweaveEmbedTransport(FakeDense(error=error))
        with pytest.raises(EmbedderRateLimitError):
            await bound_call(transport, ["Acme"])

    async def test_deadline_stops_the_wait_and_cancels_the_call(self):
        dense = FakeDense(delay=5)
        transport = AirweaveEmbedTransport(dense)
        with pytest.raises(TimeoutError):
            await bound_call(transport, ["Acme"], timeout=0.1)
        await asyncio.sleep(0.05)
        assert not [t for t in asyncio.all_tasks() if "embed_many" in repr(t.get_coro())]

    async def test_unbound_transport_refuses(self):
        with pytest.raises(RuntimeError, match="bind"):
            await run_on_thread(AirweaveEmbedTransport(FakeDense()), ["Acme"])

    async def test_binding_is_per_thread(self):
        loop = asyncio.get_running_loop()
        transport = AirweaveEmbedTransport(FakeDense())
        mine = ThreadPoolExecutor(max_workers=1)
        other = ThreadPoolExecutor(max_workers=1)
        await loop.run_in_executor(mine, transport.bind, loop, 5)
        vectors, _ = await loop.run_in_executor(mine, transport, ["Acme"])
        assert len(vectors) == 1
        with pytest.raises(RuntimeError, match="bind"):
            await loop.run_in_executor(other, transport, ["Acme"])
        mine.shutdown()
        other.shutdown()


class TestMatches:
    @pytest.mark.parametrize(
        "model, dims, expected",
        [
            ("text-embedding-3-small", 1536, True),
            ("text-embedding-3-small", 1024, False),
            ("text-embedding-3-large", 1536, False),
            (None, None, False),
        ],
    )
    def test_model_and_dimensions_must_both_agree(self, model, dims, expected):
        dense = FakeDense("text-embedding-3-small", 1536)
        config = {"embedding": {} if model is None else {"model": model, "dimensions": dims}}
        assert matches(dense, config) is expected

    def test_no_embedder_never_matches(self):
        assert matches(None, {"embedding": {"model": "x", "dimensions": 1}}) is False


# --- the pool's choice of transport -----------------------------------------------------------


class Recorder:
    created: list = []

    def __init__(self, taxonomy_path, config, transport=None):
        self.transport = transport
        self.taxonomy = types.SimpleNamespace()
        self.embedder = types.SimpleNamespace(cache=types.SimpleNamespace(close=lambda: None))
        Recorder.created.append(self)

    def classify(self, rows):
        if self.transport is not None:
            self.transport(["probe"])
        return types.SimpleNamespace(enriched=[{"chunk_id": r["chunk_id"]} for r in rows])


@pytest.fixture
def stand_in(monkeypatch):
    Recorder.created = []
    integ = types.ModuleType("taxonomy_classifier.integration.airweave")
    integ.TaxonomyClassifier = Recorder
    contract = types.ModuleType("taxonomy_classifier.contract")
    contract.to_contract = lambda e, h: dict(e)
    for name, mod in (
        ("taxonomy_classifier", types.ModuleType("taxonomy_classifier")),
        ("taxonomy_classifier.integration", types.ModuleType("x")),
        ("taxonomy_classifier.integration.airweave", integ),
        ("taxonomy_classifier.contract", contract),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def taxonomy(model, dims, sha="a" * 64, version="acme-1"):
    return ResolvedTaxonomy(
        "neo-acme",
        "t-acme",
        version,
        sha,
        1,
        "calibrated",
        None,
        Path("/tmp/none"),
        {"embedding": {"model": model, "dimensions": dims}},
        RegistryMap("acme-1", {}, {}),
    )


ROWS = [{"chunk_id": "c0", "document_id": "d", "text": "Acme"}]


class TestPoolChoice:
    async def test_matching_model_uses_airweaves_embedder(self, tmp_path, stand_in):
        dense = FakeDense()
        pool = ClassifierPool(1, tmp_path, dense_embedder=dense)
        await pool.classify(taxonomy("text-embedding-3-small", 4), ROWS, 10)
        assert isinstance(Recorder.created[0].transport, AirweaveEmbedTransport)
        assert dense.calls == [["probe"]]
        pool.close()

    async def test_other_model_keeps_the_packages_client(self, tmp_path, stand_in):
        dense = FakeDense()
        pool = ClassifierPool(1, tmp_path, dense_embedder=dense)
        await pool.classify(taxonomy("text-embedding-3-large", 4), ROWS, 10)
        assert Recorder.created[0].transport is None
        assert dense.calls == []
        pool.close()

    async def test_no_embedder_keeps_the_packages_client(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path)
        await pool.classify(taxonomy("text-embedding-3-small", 4), ROWS, 10)
        assert Recorder.created[0].transport is None
        pool.close()

    async def test_explicit_transport_wins_over_the_embedder(self, tmp_path, stand_in):
        def fake(batch):
            return [np.zeros(4, np.float32) for _ in batch], 0

        dense = FakeDense()
        pool = ClassifierPool(1, tmp_path, embed_transport=fake, dense_embedder=dense)
        await pool.classify(taxonomy("text-embedding-3-small", 4), ROWS, 10)
        assert Recorder.created[0].transport is fake
        pool.close()

    async def test_each_call_binds_its_own_deadline_and_loop(self, tmp_path, stand_in):
        dense = FakeDense(delay=2)
        pool = ClassifierPool(1, tmp_path, dense_embedder=dense)
        with pytest.raises(asyncio.TimeoutError):
            await pool.classify(taxonomy("text-embedding-3-small", 4), ROWS, 0.2)
        await asyncio.sleep(0.4)  # the bridge gives up at the same deadline and cancels the call
        pool.close()


class TestFallbackIsVisible:
    """A taxonomy built with another model keeps the package's client: say so, once, and count."""

    async def test_warns_once_per_version_with_both_model_names(self, tmp_path, stand_in, caplog):
        tally = TaxonomyTally()
        pool = ClassifierPool(2, tmp_path, dense_embedder=FakeDense(), tally=tally)
        other = taxonomy("text-embedding-3-large", 3072, version="acme-2")
        with caplog.at_level("WARNING"):
            for _ in range(4):  # two threads, repeated batches
                await pool.classify(other, ROWS, 10)
        messages = [r.getMessage() for r in caplog.records]
        warnings = [m for m in messages if "package's own OpenAI client" in m]
        assert len(warnings) == 1
        for needle in ("acme-2", "text-embedding-3-large", "3072", "text-embedding-3-small", "4"):
            assert needle in warnings[0]
        assert tally.embed_fallbacks() == 1
        pool.close()

    async def test_each_version_warns_and_counts_on_its_own(self, tmp_path, stand_in, caplog):
        tally = TaxonomyTally()
        pool = ClassifierPool(1, tmp_path, dense_embedder=FakeDense(), tally=tally)
        with caplog.at_level("WARNING"):
            await pool.classify(taxonomy("other", 4, version="v1"), ROWS, 10)
            await pool.classify(taxonomy("other", 4, version="v2"), ROWS, 10)
        assert tally.embed_fallbacks() == 2
        pool.close()

    async def test_matching_taxonomy_is_silent(self, tmp_path, stand_in, caplog):
        tally = TaxonomyTally()
        pool = ClassifierPool(1, tmp_path, dense_embedder=FakeDense(), tally=tally)
        with caplog.at_level("WARNING"):
            await pool.classify(taxonomy("text-embedding-3-small", 4), ROWS, 10)
        assert not [r for r in caplog.records if "package's own OpenAI client" in r.getMessage()]
        assert tally.embed_fallbacks() == 0
        pool.close()

    async def test_dimensions_alone_trigger_it(self, tmp_path, stand_in, caplog):
        tally = TaxonomyTally()
        pool = ClassifierPool(1, tmp_path, dense_embedder=FakeDense(), tally=tally)
        with caplog.at_level("WARNING"):
            await pool.classify(taxonomy("text-embedding-3-small", 1024), ROWS, 10)
        assert tally.embed_fallbacks() == 1
        pool.close()

    async def test_explicit_transport_or_no_embedder_is_not_a_fallback(self, tmp_path, stand_in):
        tally = TaxonomyTally()
        no_embedder = ClassifierPool(1, tmp_path, tally=tally)
        await no_embedder.classify(taxonomy("other", 4), ROWS, 10)
        explicit = ClassifierPool(1, tmp_path, embed_transport=lambda b: ([], 0), tally=tally)
        await explicit.classify(taxonomy("other", 4, sha="b" * 64), ROWS, 10)
        assert tally.embed_fallbacks() == 0
        no_embedder.close()
        explicit.close()

    def test_exposed_as_a_prometheus_counter(self):
        tally = TaxonomyTally()
        tally.note_embed_fallback("acme-2")
        families = {f.name: f for f in TaxonomyChunksCollector(tally).collect()}
        sample = families["taxonomy_embed_fallback"].samples[0]
        assert sample.name == "taxonomy_embed_fallback_total" and sample.value == 1.0


class TestBatchDeadline:
    """A batch the caller gave up on makes no further embedding calls."""

    async def test_a_second_call_after_the_batch_deadline_is_refused_without_calling_the_embedder(
        self,
    ):
        dense = FakeDense(delay=0.0)
        transport = AirweaveEmbedTransport(dense)
        loop = asyncio.get_running_loop()

        def work():
            transport.bind(loop, 0.2)
            transport(["first"])  # within the deadline
            import time

            time.sleep(0.3)  # the batch ran out of time (the caller has marked it ERROR)
            transport(["second"])

        with pytest.raises(TimeoutError, match="deadline"):
            await loop.run_in_executor(None, work)

        assert dense.calls == [["first"]]

    async def test_one_call_is_cut_at_the_time_left_not_the_whole_timeout(self):
        dense = FakeDense(delay=2.0)
        transport = AirweaveEmbedTransport(dense)
        loop = asyncio.get_running_loop()

        def work():
            transport.bind(loop, 0.2)
            transport(["slow"])

        started = loop.time()
        with pytest.raises(TimeoutError):
            await loop.run_in_executor(None, work)
        assert loop.time() - started < 1.0

    async def test_binding_again_gives_the_next_batch_a_fresh_deadline(self):
        dense = FakeDense()
        transport = AirweaveEmbedTransport(dense)
        loop = asyncio.get_running_loop()

        def work():
            import time

            transport.bind(loop, 0.1)
            time.sleep(0.15)
            transport.bind(loop, 5)  # the next batch on this thread
            return transport(["again"])

        vectors, _ = await loop.run_in_executor(None, work)
        assert len(vectors) == 1
