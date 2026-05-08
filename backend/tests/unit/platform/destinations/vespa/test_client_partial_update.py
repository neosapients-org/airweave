"""Unit tests for VespaClient partial update and scroll helpers."""

import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from airweave.platform.destinations.vespa.client import VespaClient


@pytest.mark.asyncio
async def test_partial_update_fields_sends_correct_request():
    """partial_update_fields must send only the provided fields."""
    client = VespaClient(app=MagicMock())
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def mock_async_client(**kwargs):
        class FakeClient:
            async def put(self, url, json=None, **kw):
                captured["url"] = url
                captured["json"] = json
                resp = MagicMock()
                resp.status_code = 200
                resp.text = ""
                return resp

        yield FakeClient()

    with patch("httpx.AsyncClient", side_effect=lambda **kw: mock_async_client(**kw)):
        await client.partial_update_fields(
            schema="file_entity",
            doc_id="file_entity_abc__chunk_0",
            fields={
                "doc_categories": ["correspondence"],
                "textual_representation": "[Categories: correspondence]\n\nHello",
            },
        )

    assert "/document/v1/airweave/file_entity/docid/" in captured["url"]
    assert "doc_categories" in captured["json"]["fields"]
    assert "textual_representation" in captured["json"]["fields"]
    assert "dense_embedding" not in captured["json"]["fields"]


@pytest.mark.asyncio
async def test_scroll_all_documents_collects_documents_across_pages_and_schemas():
    """scroll_all_documents must aggregate paginated file/email documents."""
    client = VespaClient(app=MagicMock())
    collection_id = UUID("12345678-1234-1234-1234-123456789abc")
    calls: list[str] = []

    responses = [
        {"documents": [{"id": "id:airweave:file_entity::doc1", "fields": {"name": "a"}}], "continuation": "next-token"},
        {"documents": [{"id": "id:airweave:file_entity::doc2", "fields": {"name": "b"}}]},
        {"documents": [{"id": "id:airweave:email_entity::doc3", "fields": {"name": "c"}}]},
    ]

    @asynccontextmanager
    async def mock_async_client(**kwargs):
        class FakeClient:
            async def get(self, url, **kw):
                calls.append(url)
                payload = responses.pop(0)
                resp = MagicMock()
                resp.status_code = 200
                resp.text = json.dumps(payload)
                resp.json = MagicMock(return_value=payload)
                return resp

        yield FakeClient()

    with patch("httpx.AsyncClient", side_effect=lambda **kw: mock_async_client(**kw)):
        docs = await client.scroll_all_documents(collection_id, page_size=1)

    assert [doc["id"] for doc in docs] == [
        "id:airweave:file_entity::doc1",
        "id:airweave:file_entity::doc2",
        "id:airweave:email_entity::doc3",
    ]
    assert any("continuation=next-token" in url for url in calls)
    assert any("/file_entity/docid" in url for url in calls)
    assert any("/email_entity/docid" in url for url in calls)
