"""Unit tests for writing detected temporal intent into search state."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from airweave.schemas.search import AirweaveTemporalConfig
from airweave.search.operations.query_interpretation import ExtractedFilters, QueryInterpretation
from airweave.search.state import SearchState


@pytest.mark.asyncio
async def test_temporal_weight_written_to_state_without_filters():
    """Temporal intent must be preserved even when no structured filters survive validation."""
    mock_provider = MagicMock()
    mock_provider.structured_output = AsyncMock(
        return_value=ExtractedFilters(
            filters=[],
            confidence=0.95,
            temporal_weight=0.7,
        )
    )
    mock_provider.llm_tokenizer = MagicMock()
    mock_provider.count_tokens = MagicMock(return_value=100)
    mock_provider.model_spec = MagicMock()
    mock_provider.model_spec.llm_model.context_window = 100000

    op = QueryInterpretation(providers=[mock_provider])

    mock_context = MagicMock()
    mock_context.query = "last conversation with John"
    mock_context.readable_collection_id = "test-collection"
    mock_context.emitter = AsyncMock()

    state = SearchState()
    ctx = MagicMock()
    ctx.logger = MagicMock()

    with patch.object(op, "_discover_fields", return_value={"slack": {"subject": "Subject"}}):
        await op.execute(mock_context, state, ctx)

    assert state.detected_temporal_config is not None
    assert isinstance(state.detected_temporal_config, AirweaveTemporalConfig)
    assert state.detected_temporal_config.weight == 0.7
    assert state.detected_temporal_config.reference_field == "updated_at"


@pytest.mark.asyncio
async def test_temporal_weight_below_confidence_threshold_is_ignored():
    """Low-confidence interpretations must not set temporal relevance."""
    mock_provider = MagicMock()
    mock_provider.structured_output = AsyncMock(
        return_value=ExtractedFilters(
            filters=[],
            confidence=0.4,
            temporal_weight=0.7,
        )
    )
    mock_provider.llm_tokenizer = MagicMock()
    mock_provider.count_tokens = MagicMock(return_value=100)
    mock_provider.model_spec = MagicMock()
    mock_provider.model_spec.llm_model.context_window = 100000

    op = QueryInterpretation(providers=[mock_provider])

    mock_context = MagicMock()
    mock_context.query = "latest update"
    mock_context.readable_collection_id = "test-collection"
    mock_context.emitter = AsyncMock()

    state = SearchState()
    ctx = MagicMock()
    ctx.logger = MagicMock()

    with patch.object(op, "_discover_fields", return_value={"slack": {"subject": "Subject"}}):
        await op.execute(mock_context, state, ctx)

    assert state.detected_temporal_config is None
