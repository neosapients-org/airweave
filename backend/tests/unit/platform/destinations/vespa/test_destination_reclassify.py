"""Unit tests for VespaDestination reclassification flow."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from airweave.platform.destinations.vespa.destination import VespaDestination


@pytest.mark.asyncio
async def test_reclassify_calls_partial_update_for_each_doc():
    """reclassify_collection_documents should update each classified document."""
    dest = VespaDestination.__new__(VespaDestination)
    dest.set_logger(MagicMock())

    mock_client = AsyncMock()
    mock_client.scroll_all_documents = AsyncMock(
        return_value=[
            {
                "id": "id:airweave:file_entity::doc1",
                "fields": {
                    "textual_representation": "Hello world",
                    "name": "test.pdf",
                    "mime_type": "application/pdf",
                },
            }
        ]
    )
    mock_client.partial_update_fields = AsyncMock()
    mock_client._parse_vespa_document_id.return_value = ("file_entity", "doc1")
    dest._client = mock_client

    collection_id = uuid4()
    with patch(
        "airweave.platform.destinations.vespa.destination.classify_text",
        new=AsyncMock(return_value=["correspondence"]),
    ):
        result = await dest.reclassify_collection_documents(collection_id)

    assert result["processed"] == 1
    assert result["skipped"] == 0
    assert result["failed"] == 0
    mock_client.partial_update_fields.assert_called_once()


@pytest.mark.asyncio
async def test_reclassify_skips_documents_without_text():
    """reclassify_collection_documents should skip docs with no text."""
    dest = VespaDestination.__new__(VespaDestination)
    dest.set_logger(MagicMock())

    mock_client = AsyncMock()
    mock_client.scroll_all_documents = AsyncMock(
        return_value=[
            {
                "id": "id:airweave:file_entity::doc1",
                "fields": {"textual_representation": "", "name": "test.pdf"},
            }
        ]
    )
    mock_client.partial_update_fields = AsyncMock()
    dest._client = mock_client

    result = await dest.reclassify_collection_documents(uuid4())

    assert result == {"processed": 0, "skipped": 1, "failed": 0}
    mock_client.partial_update_fields.assert_not_called()
