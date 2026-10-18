"""The taxonomy step with the real classifier package (plan A7 "done when").

Needs ``taxonomy_classifier`` (``feature/validated-assignment-lifecycle``) installed; skipped
otherwise. Everything is offline: taxonomies built by the package's engine with a bag-of-words
fake embedder and a fake reviewer, a filesystem "S3", the real resolver, pool, SQLite embedding
cache and classifier, and chunks made by the real chunk processor (sentence_pack, header on).
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("taxonomy_classifier")
pytest.importorskip("sklearn")

from airweave.adapters.storage.filesystem import FilesystemBackend  # noqa: E402
from airweave.core.config import settings  # noqa: E402
from airweave.domains.converters.fakes.registry import FakeConverterRegistry  # noqa: E402
from airweave.domains.sync_pipeline.processors.chunk_embed import ChunkEmbedProcessor  # noqa: E402
from airweave.domains.sync_pipeline.processors.taxonomy import TaxonomyProcessor  # noqa: E402
from airweave.domains.taxonomy.artifacts import TaxonomyResolver  # noqa: E402
from airweave.domains.taxonomy.classifier_pool import ClassifierPool  # noqa: E402
from airweave.domains.taxonomy.metrics import TaxonomyTally  # noqa: E402
from airweave.domains.taxonomy.tests.taxonomy_fixtures import (  # noqa: E402
    CORPORA,
    DIM,
    BowEmbeddings,
    bow_vector,
    build_taxonomy,
    publish,
    sentence,
    write_model,
)
from airweave.platform.destinations.vespa.transformer import EntityTransformer  # noqa: E402
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb  # noqa: E402
from airweave.platform.entities.stub import PdfStubFileEntity  # noqa: E402

pytestmark = pytest.mark.slow
LABEL_FIELDS = ("taxonomy_status", "topic_ids", "broad_family_ids", "taxonomy_spans",
                "taxonomy_version", "taxonomy_piece_count", "taxonomy_admin_pieces",
                "taxonomy_unresolved_pieces")
BUILD_ID = re.compile(r"\b(CAT\d+|CAT_NEW_\d+|BF\d+)\b")
LABELLED = {"ASSIGNED", "PARTIALLY_ASSIGNED"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Two tenants' taxonomies, published in every variant the tests need."""
    tmp = tmp_path_factory.mktemp("a7")
    s3 = tmp / "s3"
    acme, acme_cfg = build_taxonomy(tmp, "acme", "acme-2026.10")
    globex, globex_cfg = build_taxonomy(tmp, "globex", "globex-2026.10")
    good, strict = tmp / "model-good", tmp / "model-strict"
    write_model(good, 0.9, model_id="acme_hgb")
    write_model(strict, 0.75, model_id="acme_strict")
    write_model(tmp / "model-schema", 0.9, schema="learned-acceptance-features/0", model_id="old")
    pub = {
        "neo-acme": publish(s3, acme, acme_cfg, "acme", "acme-2026.10", ["neo-acme"]),
        "neo-globex": publish(s3, globex, globex_cfg, "globex", "globex-2026.10", ["neo-globex"]),
        "neo-acme-learned": publish(s3, acme, acme_cfg, "acme", "acme-2026.10-l", ["neo-acme-learned"],
                                    policy="learned", model_dir=good, model_id="acme_hgb"),
        "neo-acme-strict": publish(s3, acme, acme_cfg, "acme", "acme-2026.10-s", ["neo-acme-strict"],
                                   policy="learned", model_dir=strict, model_id="acme_strict"),
        "neo-acme-schema": publish(s3, acme, acme_cfg, "acme", "acme-2026.10-x", ["neo-acme-schema"],
                                   policy="learned", model_dir=tmp / "model-schema", model_id="old"),
        "neo-acme-nomodel": publish(s3, acme, acme_cfg, "acme", "acme-2026.10-n", ["neo-acme-nomodel"],
                                    policy="learned", model_dir=good, model_id="acme_gone"),
        "neo-acme-badsha": publish(s3, acme, acme_cfg, "acme", "acme-2026.10-b", ["neo-acme-badsha"],
                                   policy="learned", model_dir=good, model_id="acme_swapped"),
    }
    first = sorted(pub["neo-acme"].registry["categories"])[0]
    pub["neo-acme-partial"] = publish(s3, acme, acme_cfg, "acme", "acme-2026.10-p", ["neo-acme-partial"],
                                      drop_categories=[first])
    (s3 / "taxonomy/models/acme_gone/frozen_model.pkl").unlink()
    (s3 / "taxonomy/models/acme_swapped/frozen_model.pkl").write_bytes(b"not the verified model")
    return {"tmp": tmp, "s3": s3, "pub": pub, "dropped": first}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "TAXONOMY_ENABLED", True)


class SpyPool(ClassifierPool):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0
        self.rows: list = []

    async def classify(self, taxonomy, rows, timeout):
        self.calls += 1
        self.rows.extend(rows)
        return await super().classify(taxonomy, rows, timeout)


def make_processor(world, tmp_path, threads=2, delay=0.0, timeout=60.0):
    resolver = TaxonomyResolver(FilesystemBackend(world["s3"]), "taxonomy/", tmp_path / "local", 300)
    pool = SpyPool(threads, tmp_path / "emb-cache", embed_transport=BowEmbeddings(delay))
    return TaxonomyProcessor(resolver, pool, timeout, TaxonomyTally()), pool


def document(tenant: str, seed: int) -> str:
    """Paragraphs of one topic each, so chunks are topical; a footer at the end."""
    import numpy as np

    rng = np.random.default_rng(seed)
    paras = []
    for topic, phrases in CORPORA[tenant].items():
        paras.append(" ".join(sentence(p) for p in rng.choice(phrases, size=8)))
    return "\n\n".join(paras) + "\n\nPage 1 of 9"


async def make_chunks(tenant: str, n_docs: int = 2, seed: int = 100) -> list:
    """Real chunk entities: sentence_pack at 120 with the header, as Neo runs it (scaled)."""
    proc = ChunkEmbedProcessor(FakeConverterRegistry(), MagicMock(), MagicMock())
    ctx = MagicMock()
    ctx.source_short_name = "google_drive"
    ents = []
    for d in range(n_docs):
        eid = f"{tenant}-doc-{seed}-{d}"
        ents.append(PdfStubFileEntity(
            entity_id=eid, name=f"{tenant.title()} report {d}.pdf", stub_id=eid,
            file_name=f"{tenant}{d}.pdf", url="stub://x", size=1, file_type="pdf",
            breadcrumbs=[Breadcrumb(entity_id="f", name="Reports", entity_type="Folder")],
            textual_representation=document(tenant, seed + d),
            airweave_system_metadata=AirweaveSystemMetadata(source_name="google_drive")))
    with pytest.MonkeyPatch.context() as mp:
        for k, v in {"CHUNKER_MODE": "sentence_pack", "CHUNK_TARGET_TOKENS": 120}.items():
            mp.setattr(settings, k, v)  # the overlap (15%, 18) and the header follow from the mode
        return await proc._chunk_textual_entities(ents, ctx)


def ctx_for(readable: str):
    ctx = MagicMock()
    ctx.collection.readable_id = readable
    ctx.sync_job_id = uuid4()
    return ctx


def assert_contract_fields(chunk, lineage: str, version: str) -> None:
    """Exactly the contract's label fields, registry ids only, spans in the own range."""
    assert all(getattr(chunk, f) is not None for f in LABEL_FIELDS)
    assert chunk.taxonomy_version == version
    assert all(k.startswith(f"{lineage}:") and isinstance(v, int) and v >= 1
               for k, v in chunk.topic_ids.items())
    assert all(f.startswith(f"{lineage}:") for f in chunk.broad_family_ids)
    for span in json.loads(chunk.taxonomy_spans):
        assert span["c"] in chunk.topic_ids
        assert chunk.chunk_own_start <= span["s"] < span["e"] <= chunk.chunk_char_end
    assert chunk.taxonomy_admin_pieces + chunk.taxonomy_unresolved_pieces <= chunk.taxonomy_piece_count
    written = json.dumps([chunk.topic_ids, chunk.broad_family_ids, chunk.taxonomy_spans])
    assert not BUILD_ID.search(written)
    doc = EntityTransformer(collection_id=UUID(int=7)).transform(chunk)
    assert {f: doc.fields[f] for f in LABEL_FIELDS} == {f: getattr(chunk, f) for f in LABEL_FIELDS}


class TestPolicies:
    async def test_calibrated(self, world, tmp_path, enabled):
        proc, pool = make_processor(world, tmp_path)
        chunks = await make_chunks("acme")
        await proc.process(chunks, ctx_for("neo-acme"))
        statuses = [c.taxonomy_status for c in chunks]
        assert sum(s in LABELLED for s in statuses) >= len(chunks) - 2, statuses
        for c in chunks:
            assert_contract_fields(c, "acme", "acme-2026.10")
        pool.close()

    async def test_learned_with_a_test_trained_model(self, world, tmp_path, enabled):
        proc, pool = make_processor(world, tmp_path)
        chunks = await make_chunks("acme")
        await proc.process(chunks, ctx_for("neo-acme-learned"))
        assert sum(c.taxonomy_status in LABELLED for c in chunks) >= len(chunks) - 2
        for c in chunks:
            assert_contract_fields(c, "acme", "acme-2026.10-l")
        pool.close()

    async def test_learned_policy_is_really_used(self, world, tmp_path, enabled):
        """A model whose P stays under its 0.80 threshold labels nothing (calibrated would)."""
        proc, pool = make_processor(world, tmp_path)
        chunks = await make_chunks("acme")
        await proc.process(chunks, ctx_for("neo-acme-strict"))
        assert not any(c.taxonomy_status in LABELLED for c in chunks)
        assert all(c.topic_ids == {} for c in chunks)
        pool.close()

    @pytest.mark.parametrize("readable", ["neo-acme-nomodel", "neo-acme-badsha", "neo-acme-schema"])
    async def test_model_problems_are_errors_never_calibrated(self, world, tmp_path, enabled,
                                                              readable):
        proc, pool = make_processor(world, tmp_path)
        chunks = await make_chunks("acme")
        await proc.process(chunks, ctx_for(readable))
        assert {c.taxonomy_status for c in chunks} == {"ERROR"}
        assert all(c.topic_ids is None for c in chunks)
        assert pool.calls == 0  # nothing was classified, by any policy
        pool.close()


class AirweaveBow:
    """A stand-in for Airweave's dense embedder that serves the same fake model as the build."""

    model_name = "fake-bow"
    dimensions = DIM

    def __init__(self):
        self.texts: list = []
        self.thread = None

    async def embed_many(self, texts):
        from airweave.domains.embedders.types import DenseEmbedding

        self.thread = threading.get_ident()
        self.texts.extend(texts)
        return [DenseEmbedding(vector=bow_vector(t).tolist()) for t in texts]


class TestAirweaveEmbedder:
    async def test_same_labels_through_airweaves_embedder(self, world, tmp_path, enabled):
        resolver = TaxonomyResolver(FilesystemBackend(world["s3"]), "taxonomy/", tmp_path / "l", 300)
        dense = AirweaveBow()
        pool = ClassifierPool(2, tmp_path / "via-airweave", dense_embedder=dense)
        proc = TaxonomyProcessor(resolver, pool, 60.0, TaxonomyTally())
        via = await make_chunks("acme")
        await proc.process(via, ctx_for("neo-acme"))
        direct_proc, direct_pool = make_processor(world, tmp_path / "direct")
        direct = await make_chunks("acme")
        await direct_proc.process(direct, ctx_for("neo-acme"))
        assert dense.texts, "the classifier never used Airweave's embedder"
        assert dense.thread == threading.get_ident(), "the embedder must run on the event loop"
        assert [c.taxonomy_status for c in via] == [c.taxonomy_status for c in direct]
        assert [c.topic_ids for c in via] == [c.topic_ids for c in direct]
        assert [c.taxonomy_spans for c in via] == [c.taxonomy_spans for c in direct]
        for c in via:
            assert_contract_fields(c, "acme", "acme-2026.10")
        pool.close()
        direct_pool.close()

    async def test_embedder_failure_is_error_and_the_sync_goes_on(self, world, tmp_path, enabled):
        class Broken(AirweaveBow):
            async def embed_many(self, texts):
                raise RuntimeError("provider down")

        resolver = TaxonomyResolver(FilesystemBackend(world["s3"]), "taxonomy/", tmp_path / "l", 300)
        pool = ClassifierPool(1, tmp_path / "c", dense_embedder=Broken())
        tally = TaxonomyTally()
        proc = TaxonomyProcessor(resolver, pool, 60.0, tally)
        chunks = await make_chunks("acme")
        out = await proc.process(chunks, ctx_for("neo-acme"))
        assert out is chunks
        assert {c.taxonomy_status for c in chunks} == {"ERROR"}
        pool.close()


class TestIds:
    async def test_untranslated_category_dropped(self, world, tmp_path, enabled):
        proc, pool = make_processor(world, tmp_path)
        chunks = await make_chunks("acme")
        await proc.process(chunks, ctx_for("neo-acme-partial"))
        dropped = world["dropped"]
        written = json.dumps([[c.topic_ids, c.taxonomy_spans] for c in chunks])
        assert dropped not in written and not BUILD_ID.search(written)
        assert sum(c.taxonomy_unresolved_pieces for c in chunks) > 0
        pool.close()

    async def test_classifier_saw_own_text_only(self, world, tmp_path, enabled):
        proc, pool = make_processor(world, tmp_path)
        chunks = await make_chunks("acme", n_docs=1)
        assert any(c.chunk_own_start > c.chunk_char_start for c in chunks)  # overlap exists
        await proc.process(chunks, ctx_for("neo-acme"))
        texts = {r["chunk_id"]: r["text"] for r in pool.rows}
        for c in chunks:
            body = c.textual_representation[c.chunk_body_offset:]
            assert texts[c.entity_id] == body[c.chunk_own_start - c.chunk_char_start:]
            assert not texts[c.entity_id].startswith("Acme report")  # no header
        pool.close()


class TestOneWorker:
    async def test_two_collections_two_taxonomies_interleaved(self, world, tmp_path, enabled):
        proc, pool = make_processor(world, tmp_path, threads=2)
        acme = [await make_chunks("acme", seed=200 + i) for i in range(3)]
        globex = [await make_chunks("globex", seed=300 + i) for i in range(3)]
        ctx_a, ctx_g = ctx_for("neo-acme"), ctx_for("neo-globex")
        jobs = []
        for a, g in zip(acme, globex):
            jobs += [proc.process(a, ctx_a), proc.process(g, ctx_g)]
        await asyncio.gather(*jobs)
        for batch in acme:
            for c in batch:
                assert c.taxonomy_version == "acme-2026.10"
                assert all(k.startswith("acme:") for k in c.topic_ids)
        for batch in globex:
            for c in batch:
                assert c.taxonomy_version == "globex-2026.10"
                assert all(k.startswith("globex:") for k in c.topic_ids)
        assert sum(c.taxonomy_status in LABELLED for b in acme + globex for c in b) > 0
        pool.close()

    async def test_real_sqlite_cache_under_concurrency(self, world, tmp_path, enabled):
        """Concurrent batches from several tasks: no ProgrammingError, nothing ERROR."""
        proc, pool = make_processor(world, tmp_path, threads=2)
        batches = [await make_chunks(t, seed=400 + i) for i in range(4) for t in ("acme", "globex")]
        ctxs = {"acme": ctx_for("neo-acme"), "globex": ctx_for("neo-globex")}

        async def worker(mine):
            for b in mine:
                tenant = "acme" if b[0].entity_id.startswith("acme") else "globex"
                await proc.process(b, ctxs[tenant])

        await asyncio.gather(*(worker(batches[i::4]) for i in range(4)))
        statuses = [c.taxonomy_status for b in batches for c in b]
        assert "ERROR" not in statuses
        assert sorted(p.name for p in (tmp_path / "emb-cache").glob("*.sqlite")) == sorted(
            f"{__import__('os').getpid()}-{i}.sqlite" for i in (0, 1))
        pool.close()

    async def test_without_dedicated_threads_sqlite_refuses(self, world, tmp_path):
        """The trap the pool avoids: a classifier used off its creating thread fails."""
        from taxonomy_classifier.integration.airweave import TaxonomyClassifier

        resolver = TaxonomyResolver(FilesystemBackend(world["s3"]), "taxonomy/", tmp_path / "l", 300)
        t = await resolver.resolve("neo-acme")
        cfg = {**t.config, "embedding": {**t.config["embedding"],
                                          "cache_path": str(tmp_path / "shared.sqlite")}}
        clf = TaxonomyClassifier(t.local_path, cfg, BowEmbeddings())
        errors = []

        def other_thread():
            try:
                clf.classify([{"chunk_id": "x", "document_id": "d",
                               "text": sentence("late loan payments") * 3}])
            except Exception as e:
                errors.append(e)

        th = threading.Thread(target=other_thread)
        th.start()
        th.join()
        assert errors and isinstance(errors[0], sqlite3.ProgrammingError)


class TestTimeout:
    async def test_timeout_is_error_counted_and_sync_goes_on(self, world, tmp_path, enabled):
        proc, pool = make_processor(world, tmp_path, threads=1, delay=1.5, timeout=0.3)
        first = await make_chunks("acme", seed=500)
        await proc.process(first, ctx_for("neo-acme"))
        assert {c.taxonomy_status for c in first} == {"ERROR"}
        assert proc._tally.totals()["ERROR"] == len(first)
        proc._timeout = 60.0
        second = await make_chunks("acme", seed=501)
        await proc.process(second, ctx_for("neo-acme"))  # queued behind the timed-out batch
        assert "ERROR" not in {c.taxonomy_status for c in second}
        pool.close()


class TestCacheRotationOnDisk:
    async def test_real_cache_starts_over_labels_unchanged(self, world, tmp_path, enabled):
        resolver = TaxonomyResolver(
            FilesystemBackend(world["s3"]), "taxonomy/", tmp_path / "l", 300
        )
        cache = tmp_path / "cache"
        pool = ClassifierPool(1, cache, embed_transport=BowEmbeddings(), max_cache_mb=0.01)
        proc = TaxonomyProcessor(resolver, pool, 60.0, TaxonomyTally())
        first = await make_chunks("acme", n_docs=3, seed=300)
        await proc.process(first, ctx_for("neo-acme"))
        grown = sorted(cache.glob("*.sqlite"))
        assert grown and grown[0].stat().st_size > 0.01 * 1024 * 1024
        second = await make_chunks("acme", n_docs=3, seed=300)
        await proc.process(second, ctx_for("neo-acme"))
        files = sorted(p.name for p in cache.glob("*.sqlite"))
        assert len(files) == 1 and "-g1" in files[0], files  # the full file was deleted
        assert [c.topic_ids for c in second] == [c.topic_ids for c in first]
        pool.close()
        assert not list(cache.glob("*.sqlite*")), "close must leave nothing on the volume"
