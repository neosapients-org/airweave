"""A Confluence page's raw HTML body must not travel with the entity once its file is saved.

The chunker deep-copies the page entity for every chunk, and every chunk's sparse embedding, Vespa
payload and raw-data copy are made from the entity's serialised fields. A body left on the entity
was therefore copied into each chunk: one 530 KB page with 140 chunks became ~74 MB held in the
sync worker (which was OOM-killed) and stored in Vespa, and every chunk's BM25 vector carried the
whole page's terms. The page's content is in the saved HTML file, which is what gets chunked.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from airweave.domains.storage import FileSkippedException
from airweave.platform.entities._base import Breadcrumb
from airweave.platform.sources.confluence import ConfluenceSource

BODY = "<p>" + "quarterly revenue forecast " * 2000 + "</p>"


def _page_detail(page_id: str = "146997253") -> dict:
    return {
        "id": page_id,
        "title": "Revenue plan",
        "status": "current",
        "createdAt": "2026-08-01T10:00:00Z",
        "version": {"number": 3},
        "space": {"id": "98307"},
        "body": {"storage": {"value": BODY}},
    }


def _source(pages: list[dict]) -> ConfluenceSource:
    source = ConfluenceSource(auth=AsyncMock(), logger=MagicMock(), http_client=AsyncMock())
    source._base_url = "https://api.atlassian.com/ex/confluence/cloud-1"
    listing = {"results": [{"id": p["id"]} for p in pages], "_links": {}}
    details = {p["id"]: p for p in pages}

    async def get(url: str):
        for page_id, detail in details.items():
            if f"/pages/{page_id}?" in url:
                return detail
        return listing

    source._get = AsyncMock(side_effect=get)
    return source


def _files(saved: list[bytes]):
    async def save(entity, content, filename_with_extension, logger):
        saved.append(content)
        entity.local_path = f"/tmp/{filename_with_extension}"
        return entity

    files = MagicMock()
    files.save_bytes = AsyncMock(side_effect=save)
    return files


async def _pages(source: ConfluenceSource, files) -> list:
    crumb = Breadcrumb(entity_id="98307", name="Finance", entity_type="ConfluenceSpaceEntity")
    return [
        e
        async for e in source._generate_page_entities(
            "98307", "FIN", crumb, "https://example.atlassian.net", files
        )
    ]


@pytest.mark.asyncio
async def test_saved_page_keeps_its_content_in_the_file_not_the_entity():
    saved: list[bytes] = []

    pages = await _pages(_source([_page_detail()]), _files(saved))

    assert len(pages) == 1
    assert BODY.encode() in saved[0]
    assert pages[0].local_path == "/tmp/Revenue plan.html"
    assert pages[0].body is None


@pytest.mark.asyncio
async def test_page_body_reaches_no_serialised_copy():
    pages = await _pages(_source([_page_detail()]), _files([]))

    dumped = pages[0].model_dump(mode="json")

    assert "quarterly revenue forecast" not in str(dumped)


@pytest.mark.asyncio
async def test_a_skipped_page_is_not_yielded():
    files = MagicMock()
    files.save_bytes = AsyncMock(side_effect=FileSkippedException("too large", "Revenue plan"))

    pages = await _pages(_source([_page_detail()]), files)

    assert pages == []
