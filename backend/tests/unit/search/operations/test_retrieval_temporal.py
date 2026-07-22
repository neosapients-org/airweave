"""Unit tests for Retrieval temporal propagation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from airweave.schemas.search import AirweaveTemporalConfig, RetrievalStrategy
from airweave.search.operations.retrieval import Retrieval
from airweave.search.state import SearchState


@pytest.mark.asyncio
async def test_retrieval_passes_temporal_config_to_destination():
    """Retrieval must forward temporal_config from context to destination.search()."""
    mock_dest = AsyncMock()
    mock_dest.search = AsyncMock(return_value=[])

    op = Retrieval(
        destination=mock_dest,
        strategy=RetrievalStrategy.HYBRID,
        offset=0,
        limit=10,
    )

    mock_context = MagicMock()
    mock_context.query = "latest email from John"
    mock_context.collection_id = "abc"
    mock_context.reranking = None
    mock_context.temporal_config = AirweaveTemporalConfig(weight=0.5)
    mock_context.emitter = AsyncMock()

    mock_state = SearchState()
    mock_state.dense_embeddings = [[0.1] * 3072]
    mock_state.sparse_embeddings = [MagicMock()]
    mock_state.filter = None
    mock_state.expanded_queries = []

    mock_ctx = MagicMock()
    mock_ctx.logger = MagicMock()

    await op.execute(mock_context, mock_state, mock_ctx)

    call_kwargs = mock_dest.search.call_args.kwargs
    assert call_kwargs["temporal_config"] is not None
    assert call_kwargs["temporal_config"].weight == 0.5


@pytest.mark.asyncio
async def test_retrieval_prefers_detected_temporal_config_over_context():
    """State-detected temporal config must override the request-level temporal config."""
    mock_dest = AsyncMock()
    mock_dest.search = AsyncMock(return_value=[])

    op = Retrieval(
        destination=mock_dest,
        strategy=RetrievalStrategy.HYBRID,
        offset=0,
        limit=10,
    )

    mock_context = MagicMock()
    mock_context.query = "latest email from John"
    mock_context.collection_id = "abc"
    mock_context.reranking = None
    mock_context.temporal_config = AirweaveTemporalConfig(weight=0.2)
    mock_context.emitter = AsyncMock()

    mock_state = SearchState()
    mock_state.dense_embeddings = [[0.1] * 3072]
    mock_state.sparse_embeddings = [MagicMock()]
    mock_state.filter = None
    mock_state.expanded_queries = []
    mock_state.detected_temporal_config = AirweaveTemporalConfig(weight=0.7)

    mock_ctx = MagicMock()
    mock_ctx.logger = MagicMock()

    await op.execute(mock_context, mock_state, mock_ctx)

    call_kwargs = mock_dest.search.call_args.kwargs
    assert call_kwargs["temporal_config"] is not None
    assert call_kwargs["temporal_config"].weight == 0.7
