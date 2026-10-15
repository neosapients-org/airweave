"""A chunk written with ``ERROR`` still counts as synced, and is not a dropped entity (batch 3).

After the merge of ``feat/document-classification`` a destination handler reports the entities
that "produced no content" and the dispatcher keeps those out of the Postgres metadata, so they
are retried at the next sync. Labelling never removes a chunk, so a chunk written without labels
because the taxonomy step failed (contract C20: fail open, counted) must stay out of that set:
its entity is recorded as synced and is not retried.

Real: ``EntityActionDispatcher``, ``DestinationHandler``, ``TaxonomyProcessor``. Faked: the
chunk/embed processor, the destination, the classifier pool and the metadata handler.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from airweave.core.config import settings
from airweave.domains.sync_pipeline.entity.actions import EntityActionBatch, EntityInsertAction
from airweave.domains.sync_pipeline.entity.dispatcher import EntityActionDispatcher
from airweave.domains.sync_pipeline.entity.handlers.destination import DestinationHandler
from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError
from airweave.domains.taxonomy.metrics import TaxonomyTally
from airweave.domains.taxonomy.tests.test_taxonomy_processor import (
    FakePool,
    FakeResolver,
    chunks,
    processor,
    sync_context,
)
from airweave.platform.entities.stub import PdfStubFileEntity

ENTITY_ID = "acme-q3"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "TAXONOMY_ENABLED", True)


class Recorder:
    """The metadata handler: records which entities it was asked to mark as synced."""

    name = "entity_postgres_metadata"

    def __init__(self):
        self.synced: list = []

    async def handle_batch(self, batch, sync_context, runtime):
        self.synced.extend(a.entity_id for a in batch.inserts + batch.updates)
        return set()


def source_entity() -> PdfStubFileEntity:
    return PdfStubFileEntity(
        entity_id=ENTITY_ID,
        name="Acme Q3.pdf",
        stub_id=ENTITY_ID,
        file_name="Acme Q3.pdf",
        url="stub://x",
        size=1,
        file_type="pdf",
        breadcrumbs=[],
    )


def build(taxonomy_step, produced=chunks):
    """Dispatcher -> destination handler (taxonomy step on) -> one fake destination."""
    written = []
    destination = MagicMock()
    destination.__class__.__name__ = "FakeVespa"
    destination.soft_fail = False
    destination.bulk_insert = AsyncMock(side_effect=lambda ents: written.extend(ents))
    destination.bulk_delete_by_parent_ids = AsyncMock()
    chunk_embed = MagicMock()
    chunk_embed.process = AsyncMock(side_effect=lambda ents, ctx, rt: produced())
    handler = DestinationHandler([destination], chunk_embed, taxonomy=taxonomy_step)
    handler._classifier = MagicMock(enabled=False)
    metadata = Recorder()
    return EntityActionDispatcher([handler], metadata), metadata, written, handler


def batch_of_one() -> EntityActionBatch:
    return EntityActionBatch(inserts=[EntityInsertAction(source_entity(), "pdf")])


def ctx():
    context = sync_context()
    context.logger = MagicMock()
    context.sync.id = "sync-1"
    return context


FAILURES = {
    "classifier raised": lambda: processor(pool=FakePool(fail=RuntimeError("boom"))),
    "classifier timed out": lambda: processor(pool=FakePool(fail=asyncio.TimeoutError())),
    "taxonomy not resolvable": lambda: processor(
        resolver=FakeResolver(TaxonomyResolutionError("checksum mismatch"))
    ),
}


@pytest.mark.parametrize("make_step", FAILURES.values(), ids=FAILURES.keys())
class TestErrorChunkIsSynced:
    async def test_chunks_are_written_with_error_and_no_labels(self, make_step):
        dispatcher, _, written, _ = build(make_step())
        await dispatcher.dispatch(batch_of_one(), ctx(), MagicMock())
        assert len(written) == 3
        assert {c.taxonomy_status for c in written} == {"ERROR"}
        assert all(not c.topic_ids and not c.taxonomy_spans for c in written)

    async def test_the_handler_reports_nothing_as_dropped(self, make_step):
        _, _, _, handler = build(make_step())
        dropped = await handler.handle_batch(batch_of_one(), ctx(), MagicMock())
        assert dropped == set()

    async def test_the_entity_is_recorded_as_synced(self, make_step):
        dispatcher, metadata, _, _ = build(make_step())
        context = ctx()
        await dispatcher.dispatch(batch_of_one(), context, MagicMock())
        assert metadata.synced == [ENTITY_ID]

    async def test_the_step_must_never_remove_or_reorder_a_chunk(self, make_step):
        # The handler works out the dropped entities BEFORE the labels are set, so the whole
        # guarantee rests on the step returning every chunk, in order, whatever went wrong.
        given = chunks()
        order = [c.entity_id for c in given]
        returned = await make_step().process(given, ctx())
        assert returned is given
        assert [c.entity_id for c in returned] == order

    async def test_no_dropped_entity_warning_is_logged(self, make_step):
        dispatcher, _, _, _ = build(make_step())
        context = ctx()
        await dispatcher.dispatch(batch_of_one(), context, MagicMock())
        warnings = " ".join(str(c) for c in context.logger.warning.call_args_list)
        assert "produced no destination" not in warnings
        assert "will not be marked synced" not in warnings


class TestNeverRemovesOrReorders:
    """The same invariant on the paths that do not fail."""

    async def test_labelled(self):
        given = chunks()
        order = [c.entity_id for c in given]
        returned = await processor().process(given, ctx())
        assert returned is given and [c.entity_id for c in returned] == order

    async def test_a_chunk_the_classifier_returns_nothing_for_is_still_returned(self):
        given = chunks()
        order = [c.entity_id for c in given]
        step = processor(pool=FakePool(drop=[given[1].entity_id]))
        returned = await step.process(given, ctx())
        assert returned is given and [c.entity_id for c in returned] == order

    async def test_step_off(self, monkeypatch):
        monkeypatch.setattr(settings, "TAXONOMY_ENABLED", False)
        given = chunks()
        returned = await processor().process(given, ctx())
        assert returned is given


class TestControls:
    """The same wiring still reports a real drop, so the checks above can fail."""

    async def test_an_entity_with_no_chunks_is_still_dropped_with_the_step_on(self):
        dispatcher, metadata, written, _ = build(processor(), produced=lambda: [])
        await dispatcher.dispatch(batch_of_one(), ctx(), MagicMock())
        assert written == [] and metadata.synced == []

    async def test_labelled_chunks_are_synced_too(self):
        dispatcher, metadata, written, _ = build(processor())
        await dispatcher.dispatch(batch_of_one(), ctx(), MagicMock())
        assert {c.taxonomy_status for c in written} == {"ASSIGNED"}
        assert metadata.synced == [ENTITY_ID]

    async def test_errors_are_counted_not_hidden(self):
        tally = TaxonomyTally()
        step = processor(pool=FakePool(fail=RuntimeError("boom")), tally=tally)
        dispatcher, _, _, _ = build(step)
        await dispatcher.dispatch(batch_of_one(), ctx(), MagicMock())
        assert tally.totals()["ERROR"] == 3
