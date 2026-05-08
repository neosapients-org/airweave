"""Unit tests for admin reclassification endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from airweave.api.v1.endpoints.admin import reclassify_collection
from airweave.domains.collections.fakes.repository import FakeCollectionRepository


@pytest.fixture
def mock_ctx():
    """Mock API context with admin permissions."""
    ctx = MagicMock()
    ctx.logger = MagicMock()
    ctx.request_id = "req-123"
    ctx.user = MagicMock()
    ctx.user.id = uuid4()
    ctx.user.is_admin = True
    ctx.user.is_superuser = True
    ctx.organization = MagicMock()
    ctx.organization.id = uuid4()
    return ctx


@pytest.fixture
def mock_db():
    """Mock database session."""
    return AsyncMock()


def _collection_repo(readable_id: str, collection_id, organization_id) -> FakeCollectionRepository:
    collection = MagicMock()
    collection.id = collection_id
    collection.organization_id = organization_id
    repo = FakeCollectionRepository()
    repo.seed_readable(readable_id, collection)
    return repo


@pytest.mark.asyncio
async def test_reclassify_endpoint_requires_admin_permission(mock_ctx, mock_db):
    """Reclassify endpoint must enforce admin permission."""
    repo = FakeCollectionRepository()

    with patch("airweave.api.v1.endpoints.admin._require_admin_permission") as mock_require:
        mock_require.side_effect = HTTPException(status_code=403, detail="forbidden")
        with pytest.raises(HTTPException) as exc_info:
            await reclassify_collection(
                readable_id="test-collection",
                db=mock_db,
                ctx=mock_ctx,
                collection_repo=repo,
            )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_reclassify_endpoint_returns_stats(mock_ctx, mock_db):
    """Reclassify endpoint must return processed/skipped/failed counts."""
    collection_id = uuid4()
    repo = _collection_repo("test-collection", collection_id, mock_ctx.organization.id)

    mock_destination = AsyncMock()
    mock_destination.reclassify_collection_documents = AsyncMock(
        return_value={"processed": 42, "skipped": 3, "failed": 0}
    )
    mock_destination.close_connection = AsyncMock()

    with patch("airweave.api.v1.endpoints.admin._require_admin_permission"):
        with patch(
            "airweave.platform.destinations.vespa.destination.VespaDestination.create",
            new=AsyncMock(return_value=mock_destination),
        ):
            response = await reclassify_collection(
                readable_id="test-collection",
                db=mock_db,
                ctx=mock_ctx,
                collection_repo=repo,
            )

    assert response.readable_collection_id == "test-collection"
    assert response.processed == 42
    assert response.skipped == 3
    assert response.failed == 0
    mock_destination.reclassify_collection_documents.assert_called_once_with(collection_id)
    mock_destination.close_connection.assert_called_once()
