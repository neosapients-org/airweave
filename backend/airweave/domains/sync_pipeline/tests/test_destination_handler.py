"""Unit tests for DestinationHandler retry and timing behavior.

Verifies that:
1. TimeoutError flows through _execute_with_retry as a retryable exception
2. After max retries, SyncFailureError is raised (fail fast, fail loud)
3. Timing logs fire for slow operations (>10s)
4. Timing logs fire for slow content processing (>10s)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from airweave.domains.sync_pipeline.exceptions import SyncFailureError
from airweave.domains.sync_pipeline.entity.handlers.destination import DestinationHandler

_ASYNC_SLEEP = "airweave.domains.sync_pipeline.entity.handlers.destination.asyncio.sleep"


def _make_mock_destination(soft_fail=False):
    """Create a mock destination with required attributes."""
    dest = MagicMock()
    dest.__class__.__name__ = "MockDestination"
    dest.soft_fail = soft_fail
    dest.bulk_insert = AsyncMock()
    dest.bulk_delete_by_parent_ids = AsyncMock()
    return dest


def _make_mock_sync_context():
    """Create a mock sync context with logger."""
    ctx = MagicMock()
    ctx.logger = MagicMock()
    ctx.logger.warning = MagicMock()
    ctx.logger.error = MagicMock()
    ctx.logger.debug = MagicMock()
    ctx.sync = MagicMock()
    ctx.sync.id = "test-sync-id"
    return ctx


class TestExecuteWithRetryTimeout:
    """Test that TimeoutError is retried and eventually fails loud."""

    @pytest.mark.asyncio
    async def test_timeout_error_is_retried(self):
        """TimeoutError should be caught and retried up to max_retries times."""
        dest = _make_mock_destination()
        handler = DestinationHandler([dest], processor=MagicMock())
        ctx = _make_mock_sync_context()

        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("feed timed out")

        with patch(_ASYNC_SLEEP, new_callable=AsyncMock):
            with pytest.raises(SyncFailureError, match="Destination unavailable"):
                await handler._execute_with_retry(
                    operation=failing_operation,
                    operation_name="insert_MockDestination",
                    destination=dest,
                    sync_context=ctx,
                    max_retries=2,
                )

        # initial attempt + 2 retries = 3 total calls
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_asyncio_timeout_error_is_retried(self):
        """asyncio.TimeoutError should use the same retry path as TimeoutError.

        From Python 3.11 onward, asyncio.TimeoutError is an alias of the builtin
        TimeoutError. On 3.10 and earlier it was a separate class, so the handler's
        retry tuple (which lists TimeoutError) does not treat it as retryable — skip
        when the two types differ.
        """
        if asyncio.TimeoutError is not TimeoutError:
            pytest.skip(
                "asyncio.TimeoutError is only an alias of TimeoutError on Python 3.11+"
            )

        dest = _make_mock_destination()
        handler = DestinationHandler([dest], processor=MagicMock())
        ctx = _make_mock_sync_context()

        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()

        with patch(_ASYNC_SLEEP, new_callable=AsyncMock):
            with pytest.raises(SyncFailureError, match="Destination unavailable"):
                await handler._execute_with_retry(
                    operation=failing_operation,
                    operation_name="insert_MockDestination",
                    destination=dest,
                    sync_context=ctx,
                    max_retries=2,
                )

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_succeeds_on_retry(self):
        """If operation succeeds on retry, no error is raised."""
        dest = _make_mock_destination()
        handler = DestinationHandler([dest], processor=MagicMock())
        ctx = _make_mock_sync_context()

        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("temporary failure")
            return "success"

        with patch(_ASYNC_SLEEP, new_callable=AsyncMock):
            result = await handler._execute_with_retry(
                operation=flaky_operation,
                operation_name="insert_MockDestination",
                destination=dest,
                sync_context=ctx,
                max_retries=4,
            )

        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_retry_logs_warning_on_each_failure(self):
        """Each retry should log a warning with attempt number."""
        dest = _make_mock_destination()
        handler = DestinationHandler([dest], processor=MagicMock())
        ctx = _make_mock_sync_context()

        async def failing_operation():
            raise TimeoutError("feed timed out")

        with patch(_ASYNC_SLEEP, new_callable=AsyncMock):
            with pytest.raises(SyncFailureError):
                await handler._execute_with_retry(
                    operation=failing_operation,
                    operation_name="insert_MockDestination",
                    destination=dest,
                    sync_context=ctx,
                    max_retries=2,
                )

        # Should have 2 warning logs (attempt 1 and 2, not the final failure)
        warning_calls = ctx.logger.warning.call_args_list
        retry_warnings = [c for c in warning_calls if "Retrying" in str(c)]
        assert len(retry_warnings) == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception_fails_immediately(self):
        """Non-retryable exceptions should fail immediately with SyncFailureError."""
        dest = _make_mock_destination()
        handler = DestinationHandler([dest], processor=MagicMock())
        ctx = _make_mock_sync_context()

        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad data")

        with pytest.raises(SyncFailureError, match="Destination failed"):
            await handler._execute_with_retry(
                operation=failing_operation,
                operation_name="insert_MockDestination",
                destination=dest,
                sync_context=ctx,
                max_retries=4,
            )

        # Should NOT retry - fails on first attempt
        assert call_count == 1


class TestTimingLogs:
    """Test that timing logs fire for slow operations."""

    @pytest.mark.asyncio
    async def test_slow_operation_logs_warning(self):
        """Operations taking >10s should log a warning."""
        dest = _make_mock_destination()
        handler = DestinationHandler([dest], processor=MagicMock())
        ctx = _make_mock_sync_context()

        # Mock the event loop time to simulate a 15-second operation
        time_values = [0.0, 15.0]  # start, end
        time_iter = iter(time_values)

        async def slow_operation():
            return "done"

        mock_loop = MagicMock()
        mock_loop.time = MagicMock(side_effect=time_iter)

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            await handler._execute_with_retry(
                operation=slow_operation,
                operation_name="insert_MockDestination",
                destination=dest,
                sync_context=ctx,
            )

        # Should log warning about slow operation
        warning_calls = ctx.logger.warning.call_args_list
        slow_warnings = [c for c in warning_calls if "slow" in str(c)]
        assert len(slow_warnings) == 1
        assert "15.0s" in str(slow_warnings[0])

    @pytest.mark.asyncio
    async def test_fast_operation_does_not_log_warning(self):
        """Operations completing in <10s should not log a warning."""
        dest = _make_mock_destination()
        handler = DestinationHandler([dest], processor=MagicMock())
        ctx = _make_mock_sync_context()

        # Mock the event loop time to simulate a 0.5-second operation
        time_values = [0.0, 0.5]
        time_iter = iter(time_values)

        async def fast_operation():
            return "done"

        mock_loop = MagicMock()
        mock_loop.time = MagicMock(side_effect=time_iter)

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            await handler._execute_with_retry(
                operation=fast_operation,
                operation_name="insert_MockDestination",
                destination=dest,
                sync_context=ctx,
            )

        # Should NOT log warning
        warning_calls = ctx.logger.warning.call_args_list
        slow_warnings = [c for c in warning_calls if "slow" in str(c)]
        assert len(slow_warnings) == 0

    @pytest.mark.asyncio
    async def test_slow_processing_logs_warning(self):
        """Content processing (chunking/embedding) taking >10s should log a warning."""
        dest = _make_mock_destination()
        mock_processor = MagicMock()
        mock_processor.__class__.__name__ = "ChunkEmbedProcessor"
        mock_processor.process = AsyncMock(return_value=[])
        handler = DestinationHandler([dest], processor=mock_processor)
        ctx = _make_mock_sync_context()

        time_values = [0.0, 15.0]
        time_iter = iter(time_values)

        mock_loop = MagicMock()
        mock_loop.time = MagicMock(side_effect=time_iter)

        mock_entity = MagicMock()
        mock_runtime = MagicMock()

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            await handler._do_process_and_insert([mock_entity], ctx, mock_runtime)

        warning_calls = ctx.logger.warning.call_args_list
        slow_warnings = [c for c in warning_calls if "slow" in str(c)]
        assert len(slow_warnings) == 1
        assert "ChunkEmbedProcessor" in str(slow_warnings[0])
        assert "15.0s" in str(slow_warnings[0])


def _make_entity(entity_id: str, mime_type: str = "application/pdf"):
    """A minimal stand-in for BaseEntity with a real (non-Mock) mime_type.

    Bare MagicMock() entities make `mime.startswith("image/")` evaluate
    truthy (an unconfigured MagicMock is truthy), which would silently route
    every entity through the image-skip branch instead of the path under
    test — so these tests use a real string here rather than a bare Mock.
    """
    entity = MagicMock()
    entity.entity_id = entity_id
    entity.mime_type = mime_type
    return entity


def _make_chunk_entity(original_entity_id: str):
    """A minimal stand-in for a post-chunking entity.

    Mirrors what ChunkEmbedProcessor._multiply_entities actually produces:
    a new entity_id of "{original}__chunk_{n}" plus original_entity_id
    stamped on airweave_system_metadata — the field _do_process_and_insert
    reads to work out which original entities survived processing.
    """
    chunk = MagicMock()
    chunk.entity_id = f"{original_entity_id}__chunk_0"
    chunk.airweave_system_metadata = MagicMock()
    chunk.airweave_system_metadata.original_entity_id = original_entity_id
    return chunk


class TestDroppedEntityReporting:
    """Regression coverage for the "Postgres marks it synced, Vespa never got it" bug.

    A conversion or embedding failure that produces zero content did not
    raise — ChunkEmbedProcessor.process() just returned fewer entities than
    it was given — so the destination handler used to report success either
    way. Postgres would then record that entity's hash as up to date even
    though nothing was ever written to any destination, and every later sync
    would see an unchanged hash and skip it forever. _do_process_and_insert
    must report exactly which entity_ids produced no content so the
    dispatcher can keep them out of what gets marked as synced.
    """

    @pytest.mark.asyncio
    async def test_returns_all_ids_when_processor_produces_nothing(self):
        """A total processing failure (e.g. OCR/conversion failure) must be
        reported for every entity that went in, not silently swallowed."""
        dest = _make_mock_destination()
        mock_processor = MagicMock()
        mock_processor.process = AsyncMock(return_value=[])
        handler = DestinationHandler([dest], processor=mock_processor)
        ctx = _make_mock_sync_context()
        runtime = MagicMock()

        entity = _make_entity("docx-1")
        dropped = await handler._do_process_and_insert([entity], ctx, runtime)

        assert dropped == {"docx-1"}
        dest.bulk_insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_set_when_everything_succeeds(self):
        """The common case: nothing dropped, no regression in behavior."""
        dest = _make_mock_destination()
        mock_processor = MagicMock()
        chunk = _make_chunk_entity("pdf-1")
        mock_processor.process = AsyncMock(return_value=[chunk])
        handler = DestinationHandler([dest], processor=mock_processor)
        ctx = _make_mock_sync_context()
        runtime = MagicMock()

        entity = _make_entity("pdf-1")
        dropped = await handler._do_process_and_insert([entity], ctx, runtime)

        assert dropped == set()
        dest.bulk_insert.assert_called_once_with([chunk])

    @pytest.mark.asyncio
    async def test_partial_failure_only_reports_the_failed_entity(self):
        """One entity fails conversion, another succeeds — only the failed
        one should come back as dropped, and the successful one must still
        be inserted."""
        dest = _make_mock_destination()
        mock_processor = MagicMock()
        surviving_chunk = _make_chunk_entity("pdf-ok")
        mock_processor.process = AsyncMock(return_value=[surviving_chunk])
        handler = DestinationHandler([dest], processor=mock_processor)
        ctx = _make_mock_sync_context()
        runtime = MagicMock()

        entities = [_make_entity("pdf-ok"), _make_entity("docx-failed")]
        dropped = await handler._do_process_and_insert(entities, ctx, runtime)

        assert dropped == {"docx-failed"}
        dest.bulk_insert.assert_called_once_with([surviving_chunk])

    @pytest.mark.asyncio
    async def test_image_only_batch_is_not_reported_as_dropped(self):
        """Images are deliberately, permanently skipped (not a failure to
        retry) — they must never show up as a dropped entity."""
        dest = _make_mock_destination()
        mock_processor = MagicMock()
        mock_processor.process = AsyncMock()
        handler = DestinationHandler([dest], processor=mock_processor)
        ctx = _make_mock_sync_context()
        runtime = MagicMock()

        entity = _make_entity("image-1", mime_type="image/png")
        dropped = await handler._do_process_and_insert([entity], ctx, runtime)

        assert dropped == set()
        mock_processor.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_batch_propagates_dropped_ids(self):
        """handle_batch must surface _do_process_and_insert's dropped ids —
        this is what the dispatcher relies on to protect Postgres."""
        from airweave.domains.sync_pipeline.entity.actions import (
            EntityActionBatch,
            EntityUpdateAction,
        )

        dest = _make_mock_destination()
        mock_processor = MagicMock()
        mock_processor.process = AsyncMock(return_value=[])
        handler = DestinationHandler([dest], processor=mock_processor)
        ctx = _make_mock_sync_context()
        runtime = MagicMock()

        entity = _make_entity("docx-1")
        action = MagicMock(spec=EntityUpdateAction)
        action.entity = entity
        action.entity_id = "docx-1"
        batch = EntityActionBatch(updates=[action])

        dropped = await handler.handle_batch(batch, ctx, runtime)

        assert dropped == {"docx-1"}
