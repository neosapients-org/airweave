"""The read-only scroll of unresolved chunks (P7 design section 6, "what persists the pieces")."""

import base64

import pytest

from airweave.domains.taxonomy.relabel.jobs import InMemoryJobRepository
from airweave.domains.taxonomy.relabel.service import RelabelRefused, RelabelService
from airweave.domains.taxonomy.relabel.tests.helpers import (
    COLLECTION,
    NEW,
    OLD,
    OTHER_COLLECTION,
    FakeChunkStore,
    FakeResolver,
    chunk,
)


def world():
    return FakeChunkStore(
        [
            chunk(
                0,
                doc="a",
                status="UNRESOLVED",
                version=NEW,
                body="Acme loan text one.",
                unresolved=1,
            ),
            chunk(1, doc="a", status="PARTIALLY_ASSIGNED", version=NEW, unresolved=1),
            chunk(2, doc="a", status="ASSIGNED", version=NEW),
            chunk(0, doc="b", status="UNRESOLVED", version=OLD, unresolved=2),
            chunk(0, doc="c", status="ERROR"),
            chunk(1, doc="c", status="ADMINISTRATIVE", version=NEW),
            chunk(0, doc="d"),
            chunk(0, doc="z", status="UNRESOLVED", version=NEW, collection=OTHER_COLLECTION),
            chunk(0, doc="code", schema="code_file_entity", status="UNRESOLVED", version=NEW),
        ]
    )


def svc(store=None):
    store = store or world()
    return RelabelService(
        InMemoryJobRepository(),
        lambda: store,
        configured_cap_usd=1.0,
        usd_per_thousand=0.006,
        resolver=FakeResolver(),
    )


async def page(service, limit=100, cursor=None, version=None):
    return await service.unresolved(COLLECTION, limit, cursor, version)


class TestWhatIsReturned:
    async def test_only_unresolved_and_partially_assigned_chunks_of_this_collection(self):
        items, _ = await page(svc())
        ids = sorted(i["chunk_id"] for i in items)
        assert ids == ["file_entity_a__chunk_0", "file_entity_a__chunk_1", "file_entity_b__chunk_0"]

    async def test_each_item_has_text_version_positions_and_document_id(self):
        items, _ = await page(svc())
        first = next(i for i in items if i["chunk_id"] == "file_entity_a__chunk_0")
        assert first["text"] == "Acme loan text one."
        assert first["document_id"] == "a" and first["taxonomy_version"] == NEW
        assert first["taxonomy_status"] == "UNRESOLVED" and first["unresolved_pieces"] == 1
        assert first["positions"] == {
            "char_start": 0,
            "own_start": 0,
            "char_end": len(first["text"]),
        }
        assert {"page_from", "page_to", "schema"} <= set(first)

    async def test_text_is_the_own_text_without_the_header(self):
        items, _ = await page(svc())
        assert not any("google_drive" in i["text"] for i in items)

    async def test_code_files_and_other_collections_never_appear(self):
        items, _ = await page(svc())
        assert not [i for i in items if "code" in i["chunk_id"] or i["document_id"] == "z"]

    async def test_filtered_by_version(self):
        items, _ = await page(svc(), version=OLD)
        assert [i["document_id"] for i in items] == ["b"]


class TestPaging:
    async def test_pages_cover_everything_once(self):
        service, seen, cursor = svc(), [], None
        for _ in range(20):
            items, cursor = await page(service, limit=1, cursor=cursor)
            seen += [i["chunk_id"] for i in items]
            if cursor is None:
                break
        assert sorted(seen) == sorted(
            ["file_entity_a__chunk_0", "file_entity_a__chunk_1", "file_entity_b__chunk_0"]
        )
        assert len(seen) == len(set(seen))

    async def test_the_cursor_is_opaque_and_url_safe(self):
        _, cursor = await page(svc(), limit=1)
        assert cursor and "=" not in cursor and "+" not in cursor and "/" not in cursor

    async def test_a_page_that_overshoots_the_limit_keeps_its_rows(self):
        class Greedy(FakeChunkStore):
            async def scroll(
                self, collection_id, fields, page_size, continuation=None, statuses=None
            ):
                return await super().scroll(
                    collection_id, fields, page_size * 5, continuation, statuses
                )

        greedy = Greedy([(k[0], k[1], v) for k, v in world().docs.items()])
        items, cursor = await page(svc(greedy), limit=1)
        assert len(items) >= 1  # nothing is lost: the cursor continues after the rows returned
        rest = [] if cursor is None else (await page(svc(greedy), limit=100, cursor=cursor))[0]
        assert len(items) + len(rest) == 3

    async def test_limit_is_bounded(self):
        for bad in (0, -1, 100000):
            with pytest.raises(RelabelRefused) as e:
                await page(svc(), limit=bad)
            assert e.value.status == 422

    async def test_a_cursor_that_is_not_ours_is_refused(self):
        with pytest.raises(RelabelRefused) as e:
            await page(svc(), cursor="not-a-cursor")
        assert e.value.status == 422 and e.value.code == "bad_cursor"
        junk = base64.urlsafe_b64encode(b"[1, 2, 3]").decode().rstrip("=")
        with pytest.raises(RelabelRefused):
            await page(svc(), cursor=junk)


class TestReadOnly:
    async def test_nothing_is_written(self):
        store = world()
        before = {k: dict(v) for k, v in store.docs.items()}
        await page(svc(store))
        assert store.updates == [] and store.docs == before and store.closed

    async def test_malformed_version_is_refused(self):
        with pytest.raises(RelabelRefused):
            await page(svc(), version="../x")
