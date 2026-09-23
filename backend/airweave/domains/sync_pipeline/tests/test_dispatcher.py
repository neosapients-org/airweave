"""Unit tests for EntityActionDispatcher.dispatch's Postgres-consistency guarantee.

Regression coverage for a real production bug: a destination handler that
silently produces zero content for an entity (a conversion or embedding
failure with no exception) used to still be treated as "succeeded", so the
metadata handler recorded that entity's hash as synced even though nothing
was ever written to any destination. Every later sync then saw an unchanged
hash and skipped the entity forever — permanently invisible, with no error
anywhere. `dispatch()` must exclude any entity a destination handler reports
as dropped from what reaches the metadata handler, while leaving the
`EntityActionBatch` the caller already holds untouched (the pipeline reuses
it afterward for tracker stats, the batch-processed event, and temp-file
cleanup, which should still reflect what was actually attempted).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from airweave.domains.sync_pipeline.entity.actions import (
    EntityActionBatch,
    EntityInsertAction,
    EntityUpdateAction,
)
from airweave.domains.sync_pipeline.entity.dispatcher import EntityActionDispatcher
from airweave.domains.sync_pipeline.exceptions import SyncFailureError


def _make_handler(name: str, dropped: set | None = None, raises: Exception | None = None):
    """A minimal EntityActionHandler test double."""
    handler = MagicMock()
    handler.name = name
    if raises is not None:
        handler.handle_batch = AsyncMock(side_effect=raises)
    else:
        handler.handle_batch = AsyncMock(return_value=dropped or set())
    return handler


def _make_sync_context():
    ctx = MagicMock()
    ctx.logger = MagicMock()
    return ctx


def _make_batch(*, inserts=None, updates=None) -> EntityActionBatch:
    return EntityActionBatch(inserts=inserts or [], updates=updates or [])


def _insert(entity_id: str) -> EntityInsertAction:
    entity = MagicMock()
    entity.entity_id = entity_id
    action = MagicMock(spec=EntityInsertAction)
    action.entity = entity
    action.entity_id = entity_id
    return action


def _update(entity_id: str) -> EntityUpdateAction:
    entity = MagicMock()
    entity.entity_id = entity_id
    action = MagicMock(spec=EntityUpdateAction)
    action.entity = entity
    action.entity_id = entity_id
    return action


class TestDispatchExcludesDroppedEntitiesFromMetadata:
    @pytest.mark.asyncio
    async def test_entity_dropped_by_a_destination_never_reaches_postgres(self):
        """The exact bug scenario: one destination handler reports an entity
        it could not persist. Postgres must never see it as an update."""
        vespa = _make_handler("destination[VespaDestination]", dropped={"docx-1"})
        postgres = _make_handler("entity_postgres_metadata")
        dispatcher = EntityActionDispatcher(
            destination_handlers=[vespa], metadata_handler=postgres
        )
        batch = _make_batch(updates=[_update("docx-1")])
        ctx = _make_sync_context()

        await dispatcher.dispatch(batch, ctx, MagicMock())

        postgres.handle_batch.assert_awaited_once()
        postgres_batch = postgres.handle_batch.call_args.args[0]
        assert postgres_batch.updates == [], (
            "an entity a destination reported as dropped must not be persisted "
            "as synced metadata — that would make the failure permanent"
        )

    @pytest.mark.asyncio
    async def test_caller_batch_is_never_mutated(self):
        """dispatch() reuses `batch` for stats/events/cleanup after it
        returns — it must build a separate object for Postgres, not filter
        the caller's batch in place."""
        vespa = _make_handler("destination[VespaDestination]", dropped={"docx-1"})
        postgres = _make_handler("entity_postgres_metadata")
        dispatcher = EntityActionDispatcher(
            destination_handlers=[vespa], metadata_handler=postgres
        )
        update_action = _update("docx-1")
        batch = _make_batch(updates=[update_action])
        ctx = _make_sync_context()

        await dispatcher.dispatch(batch, ctx, MagicMock())

        assert batch.updates == [update_action], (
            "the caller's batch must be untouched even though Postgres "
            "received a filtered copy"
        )

    @pytest.mark.asyncio
    async def test_fully_successful_batch_reaches_postgres_unfiltered(self):
        """No regression: when nothing is dropped, Postgres gets exactly
        what was resolved, same as before this fix."""
        vespa = _make_handler("destination[VespaDestination]", dropped=set())
        postgres = _make_handler("entity_postgres_metadata")
        dispatcher = EntityActionDispatcher(
            destination_handlers=[vespa], metadata_handler=postgres
        )
        insert_action = _insert("pdf-1")
        batch = _make_batch(inserts=[insert_action])
        ctx = _make_sync_context()

        await dispatcher.dispatch(batch, ctx, MagicMock())

        postgres_batch = postgres.handle_batch.call_args.args[0]
        assert postgres_batch.inserts == [insert_action]

    @pytest.mark.asyncio
    async def test_partial_drop_across_two_destinations_unions_correctly(self):
        """If Vespa drops one entity and ARF drops a different one, both
        must be excluded — a "synced" entity must be intact everywhere,
        not just in whichever destination happened to succeed for it."""
        vespa = _make_handler("destination[VespaDestination]", dropped={"a"})
        arf = _make_handler("arf", dropped={"b"})
        postgres = _make_handler("entity_postgres_metadata")
        dispatcher = EntityActionDispatcher(
            destination_handlers=[vespa, arf], metadata_handler=postgres
        )
        batch = _make_batch(inserts=[_insert("a"), _insert("b"), _insert("c")])
        ctx = _make_sync_context()

        await dispatcher.dispatch(batch, ctx, MagicMock())

        postgres_batch = postgres.handle_batch.call_args.args[0]
        assert [a.entity_id for a in postgres_batch.inserts] == ["c"]

    @pytest.mark.asyncio
    async def test_any_handler_failure_still_skips_postgres_entirely(self):
        """Existing all-or-nothing contract must survive this change: a raised
        SyncFailureError from any destination handler must still abort the
        whole dispatch before Postgres is ever called."""
        vespa = _make_handler(
            "destination[VespaDestination]", raises=SyncFailureError("boom")
        )
        postgres = _make_handler("entity_postgres_metadata")
        dispatcher = EntityActionDispatcher(
            destination_handlers=[vespa], metadata_handler=postgres
        )
        batch = _make_batch(inserts=[_insert("a")])
        ctx = _make_sync_context()

        with pytest.raises(SyncFailureError):
            await dispatcher.dispatch(batch, ctx, MagicMock())

        postgres.handle_batch.assert_not_awaited()
