"""The taxonomy step in the sync (plan A7 processor; contract C11, C16): fail-open, counted.

Fake resolver and pool; the real classifier runs in ``test_classifier_integration.py``.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from airweave.core.config import settings
from airweave.domains.sync_pipeline.entity.handlers.destination import DestinationHandler
from airweave.domains.sync_pipeline.processors.taxonomy import TaxonomyProcessor
from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy, TaxonomyResolutionError
from airweave.domains.taxonomy.mapping import RegistryMap
from airweave.domains.taxonomy.metrics import (
    TaxonomyChunksCollector,
    TaxonomyTally,
    log_sync_summary,
)
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb, CodeFileEntity
from airweave.platform.entities.stub import PdfStubFileEntity

DOC = (
    "Acme overview of the quarter. Late loan payments rose sharply this quarter. "
    "Globex revenue grew in every region."
)
HEADER = "Acme Q3.pdf · google_drive · Finance"
REGISTRY = RegistryMap(
    "acme-2026.10",
    {"CAT001": "acme:credit.loan_quality", "CAT002": "acme:revenue.growth"},
    {"BF01": "acme:credit"},
)
TAXONOMY = ResolvedTaxonomy(
    "neo-acme",
    "t-acme",
    "acme-2026.10",
    "a" * 64,
    3,
    "calibrated",
    None,
    Path("/tmp/none"),
    {},
    REGISTRY,
)
LABEL_FIELDS = (
    "topic_ids",
    "broad_family_ids",
    "taxonomy_spans",
    "taxonomy_version",
    "taxonomy_piece_count",
    "taxonomy_admin_pieces",
    "taxonomy_unresolved_pieces",
)


def chunk(i: int, start: int, own: int, end: int, header: bool = True) -> PdfStubFileEntity:
    body = DOC[start:end]
    e = PdfStubFileEntity(
        entity_id=f"acme-q3__chunk_{i}",
        name="Acme Q3.pdf",
        stub_id="acme-q3",
        file_name="Acme Q3.pdf",
        url="stub://x",
        size=1,
        file_type="pdf",
        breadcrumbs=[Breadcrumb(entity_id="f", name="Finance", entity_type="F")],
        textual_representation=f"{HEADER}\n{body}" if header else body,
        airweave_system_metadata=AirweaveSystemMetadata(
            original_entity_id="acme-q3", chunk_index=i
        ),
    )
    e.chunk_char_start, e.chunk_own_start, e.chunk_char_end = start, own, end
    e.chunk_body_offset = len(HEADER) + 1 if header else 0
    return e


def chunks():
    return [chunk(0, 0, 0, 30), chunk(1, 0, 30, 76), chunk(2, 56, 76, len(DOC))]


class FakeResolver:
    def __init__(self, result=TAXONOMY):
        self.result = result
        self.calls = []

    async def resolve(self, readable_id):
        self.calls.append(readable_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakePool:
    """Labels every row with CAT001 over its whole text, or fails as told."""

    def __init__(self, fail=None, drop=()):
        self.fail, self.drop = fail, set(drop)
        self.rows = []

    async def classify(self, taxonomy, rows, timeout):
        self.rows.extend(rows)
        if self.fail is not None:
            raise self.fail
        return [
            {
                "chunk_id": r["chunk_id"],
                "status": "ASSIGNED",
                "topics": [
                    {"category_id": "CAT001", "broad_family_id": "BF01", "supporting_children": 1}
                ],
                "taxonomy_children": [
                    {
                        "taxonomy_chunk_id": "t",
                        "status": "ASSIGNED",
                        "char_start": 0,
                        "char_end": len(r["text"]),
                        "topic_ids": ["CAT001"],
                    }
                ],
            }
            for r in rows
            if r["chunk_id"] not in self.drop
        ]


def sync_context(job="9e7d1c2a-0000-4000-8000-000000000002", readable="neo-acme"):
    ctx = MagicMock()
    ctx.collection.readable_id = readable
    ctx.sync_job_id = UUID(job)
    return ctx


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "TAXONOMY_ENABLED", True)


def processor(resolver=None, pool=None, tally=None):
    return TaxonomyProcessor(
        resolver or FakeResolver(), pool or FakePool(), 5.0, tally or TaxonomyTally()
    )


class TestLabelled:
    async def test_own_text_only_and_contract_fields(self, enabled):
        pool = FakePool()
        cs = chunks()
        await processor(pool=pool).process(cs, sync_context())
        assert [r["text"] for r in pool.rows] == [DOC[0:30], DOC[30:76], DOC[76:]]
        # the document is identified by sync, schema and source id, not the bare source id
        assert [r["document_id"] for r in pool.rows] == ["/PdfStubFileEntity/acme-q3"] * 3
        for c in cs:
            assert c.taxonomy_status == "ASSIGNED" and c.taxonomy_version == "acme-2026.10"
            assert c.topic_ids == {"acme:credit.loan_quality": 1}
            assert c.broad_family_ids == ["acme:credit"]
            (span,) = json.loads(c.taxonomy_spans)
            assert (span["s"], span["e"]) == (c.chunk_own_start, c.chunk_char_end)
            assert (
                c.taxonomy_piece_count,
                c.taxonomy_admin_pieces,
                c.taxonomy_unresolved_pieces,
            ) == (1, 0, 0)

    async def test_disabled_leaves_chunks_untouched(self):
        cs = chunks()
        out = await processor().process(cs, sync_context())
        assert out is cs and all(c.taxonomy_status is None for c in cs)

    async def test_code_files_untouched(self, enabled):
        code = CodeFileEntity.model_construct(entity_id="c1", textual_representation="x = 1")
        cs = chunks() + [code]
        await processor().process(cs, sync_context())
        assert code.taxonomy_status is None and cs[0].taxonomy_status == "ASSIGNED"

    async def test_chunks_without_positions_not_labelled(self, enabled):
        c = chunk(0, 0, 0, 30)
        c.chunk_char_start = c.chunk_own_start = c.chunk_char_end = None
        await processor().process([c], sync_context())
        assert c.taxonomy_status == "NOT_LABELLED" and c.topic_ids is None

    async def test_blank_own_text_is_administrative(self, enabled):
        c = chunk(0, 0, 0, 30)
        c.textual_representation = f"{HEADER}\n" + " " * 30
        pool = FakePool()
        await processor(pool=pool).process([c], sync_context())
        assert pool.rows == [] and c.taxonomy_status == "ADMINISTRATIVE"
        assert c.taxonomy_version == "acme-2026.10" and c.topic_ids == {}

    async def test_no_active_file(self, enabled):
        cs = chunks()
        await processor(resolver=FakeResolver(None)).process(cs, sync_context())
        assert {c.taxonomy_status for c in cs} == {"NOT_LABELLED"}


class TestFailOpen:
    @pytest.mark.parametrize(
        "resolver,pool",
        [
            (FakeResolver(TaxonomyResolutionError("checksum mismatch")), FakePool()),
            (FakeResolver(), FakePool(fail=RuntimeError("classifier blew up"))),
            (FakeResolver(), FakePool(fail=asyncio.TimeoutError())),
            (FakeResolver(RuntimeError("unexpected")), FakePool()),
        ],
        ids=["resolution", "classifier", "timeout", "unexpected"],
    )
    async def test_error_written_counted_and_sync_continues(self, enabled, resolver, pool):
        tally = TaxonomyTally()
        cs = chunks()
        ctx = sync_context()
        out = await processor(resolver, pool, tally).process(cs, ctx)
        assert out is cs
        for c in cs:
            assert c.taxonomy_status == "ERROR"
            assert all(getattr(c, f) is None for f in LABEL_FIELDS)
        assert tally.totals()["ERROR"] == 3
        assert ctx.logger.error.called

    async def test_missing_record_is_error_for_that_chunk(self, enabled):
        cs = chunks()
        await processor(pool=FakePool(drop={"acme-q3__chunk_1"})).process(cs, sync_context())
        assert [c.taxonomy_status for c in cs] == ["ASSIGNED", "ERROR", "ASSIGNED"]

    async def test_untranslated_id_logged(self, enabled):
        resolver = FakeResolver(
            TAXONOMY.__class__(
                **{**TAXONOMY.__dict__, "registry": RegistryMap("acme-2026.10", {}, {})}
            )
        )
        cs = chunks()
        ctx = sync_context()
        await processor(resolver=resolver).process(cs, ctx)
        assert {c.taxonomy_status for c in cs} == {"UNRESOLVED"}
        assert all(c.taxonomy_unresolved_pieces == 1 for c in cs)
        assert any("CAT001" in str(a) for a in ctx.logger.error.call_args_list)


class TestCountsAndAlert:
    async def test_alert_once_when_error_passes_one_percent(self, enabled):
        tally = TaxonomyTally()
        ctx = sync_context()
        good = processor(tally=tally)
        bad = processor(pool=FakePool(fail=RuntimeError("down")), tally=tally)
        for _ in range(40):
            await good.process(chunks(), ctx)  # 120 labelled
        assert not any("passed 1%" in str(a) for a in ctx.logger.error.call_args_list)
        await bad.process(chunks(), ctx)  # 3 of 123: 2.4%
        await bad.process(chunks(), ctx)
        alerts = [a for a in ctx.logger.error.call_args_list if "passed 1%" in str(a)]
        assert len(alerts) == 1

    async def test_sync_summary_and_metric(self, enabled):
        tally = TaxonomyTally()
        ctx = sync_context()
        await processor(tally=tally).process(chunks(), ctx)
        await processor(pool=FakePool(fail=RuntimeError("x")), tally=tally).process(chunks(), ctx)
        logger = MagicMock()
        log_sync_summary(str(ctx.sync_job_id), logger, tally)
        (msg,), _ = logger.error.call_args
        assert "6 chunks" in msg and "ASSIGNED=3" in msg and "ERROR=3" in msg
        assert "taxonomy acme-2026.10 (generation 3)" in msg
        log_sync_summary(str(ctx.sync_job_id), logger, tally)  # popped: nothing twice
        assert logger.error.call_count == 1 and not logger.info.called

        registry = CollectorRegistry()
        registry.register(TaxonomyChunksCollector(tally))
        text = generate_latest(registry).decode()
        assert 'taxonomy_chunks_total{status="ASSIGNED"} 3.0' in text
        assert 'taxonomy_chunks_total{status="ERROR"} 3.0' in text

    def test_summary_silent_without_the_step(self):
        logger = MagicMock()
        log_sync_summary("never-ran", logger, TaxonomyTally())
        assert not logger.info.called and not logger.error.called


class TestDestinationHandler:
    async def test_labels_written_before_the_insert(self, enabled):
        written = []
        dest = MagicMock()
        dest.bulk_insert = AsyncMock(
            side_effect=lambda ents: written.extend((e.entity_id, e.taxonomy_status) for e in ents)
        )
        chunk_embed = MagicMock()
        chunk_embed.process = AsyncMock(side_effect=lambda ents, ctx, rt: chunks())
        handler = DestinationHandler([dest], chunk_embed, taxonomy=processor())
        handler._classifier = MagicMock(enabled=False)
        await handler._do_process_and_insert([chunk(0, 0, 0, 30)], sync_context(), MagicMock())
        assert {s for _, s in written} == {"ASSIGNED"}

    async def test_without_the_step_writes_as_upstream(self):
        written = []
        dest = MagicMock()
        dest.bulk_insert = AsyncMock(side_effect=lambda ents: written.extend(ents))
        chunk_embed = MagicMock()
        chunk_embed.process = AsyncMock(side_effect=lambda ents, ctx, rt: chunks())
        handler = DestinationHandler([dest], chunk_embed)
        handler._classifier = MagicMock(enabled=False)
        await handler._do_process_and_insert([chunk(0, 0, 0, 30)], sync_context(), MagicMock())
        assert written and all(e.taxonomy_status is None for e in written)


class TestDocumentIdentityForTheClassifier:
    def test_equal_source_ids_in_two_syncs_or_schemas_are_different_documents(self):
        from airweave.domains.sync_pipeline.processors.taxonomy import _document_id

        a, b, c = chunk(0, 0, 0, 30), chunk(0, 0, 0, 30), chunk(0, 0, 0, 30)
        a.airweave_system_metadata.sync_id = UUID("5c0b9a6e-0000-4000-8000-000000000001")
        b.airweave_system_metadata.sync_id = UUID("5c0b9a6e-0000-4000-8000-000000000002")
        c.airweave_system_metadata.sync_id = a.airweave_system_metadata.sync_id
        c.airweave_system_metadata.entity_type = "MailEntity"

        assert len({_document_id(a), _document_id(b), _document_id(c)}) == 3
        assert _document_id(a).startswith("5c0b9a6e-0000-4000-8000-000000000001/")

    def test_chunks_of_one_document_share_one_id(self):
        from airweave.domains.sync_pipeline.processors.taxonomy import _document_id

        first, second = chunk(0, 0, 0, 30), chunk(1, 0, 30, 76)
        assert _document_id(first) == _document_id(second)
