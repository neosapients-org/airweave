"""Chunk statistics against a target version (the platform's ``chunkStats`` port)."""

from uuid import UUID

from airweave.domains.taxonomy.relabel.stats import collect_stats
from airweave.domains.taxonomy.relabel.tests.helpers import (
    COLLECTION,
    NEW,
    OLD,
    OTHER_COLLECTION,
    FakeChunkStore,
    chunk,
)


def world():
    return FakeChunkStore(
        [
            # doc-a: three chunks at the old version
            chunk(0, doc="doc-a", status="ASSIGNED", version=OLD, pieces=2, unresolved=0),
            chunk(1, doc="doc-a", status="PARTIALLY_ASSIGNED", version=OLD, pieces=3, unresolved=1),
            chunk(2, doc="doc-a", status="UNRESOLVED", version=OLD, pieces=1, unresolved=1),
            # doc-b: at the target
            chunk(0, doc="doc-b", status="ASSIGNED", version=NEW, pieces=2),
            chunk(1, doc="doc-b", status="ADMINISTRATIVE", version=NEW, pieces=1),
            # doc-c: never labelled, and an error
            chunk(0, doc="doc-c"),
            chunk(1, doc="doc-c", status="NOT_LABELLED"),
            chunk(2, doc="doc-c", status="ERROR"),
            # cannot carry labels: a code file and an unpositioned text chunk
            chunk(0, doc="doc-d", schema="code_file_entity"),
            chunk(0, doc="doc-e", positions=False),
            # another collection
            chunk(0, doc="doc-z", status="ASSIGNED", version=OLD, collection=OTHER_COLLECTION),
        ]
    )


async def stats(target=NEW, store=None):
    return await collect_stats(store or world(), COLLECTION, target, page_size=3)


class TestCounts:
    async def test_every_field(self):
        s = await stats()
        assert s.to_port() == {
            "collectionId": str(COLLECTION),
            "total": 10,
            "otherVersion": 6,  # doc-a 3, doc-c 3; code and unpositioned never count
            "incrementalBacklog": 5,  # PARTIALLY, UNRESOLVED, NOT_LABELLED x2, ERROR
            "labelled": 3,  # ASSIGNED x2, PARTIALLY_ASSIGNED
            "errorChunks": 1,
            "pieces": 9,  # 2+3+1 (doc-a) and 2+1 (doc-b); the ERROR chunk carries no counts
            "unresolvedPieces": 2,
            "documentsAtVersion": 1,
            "notLabelledChunks": 2,
            "notLabelledDocuments": 1,
            "documents": 5,  # a to e; the other collection's doc-z is not ours
            "documentsBySource": [{"source": "unknown", "documents": 5}],
            "labellableChunks": 8,
            "stampableChunks": 1,
        }

    async def test_total_counts_every_chunk_but_other_version_only_labellable_ones(self):
        s = await stats()
        assert s.total == 10 and s.labellable == 8

    async def test_scoped_to_the_collection(self):
        s = await stats(store=FakeChunkStore([chunk(0, collection=OTHER_COLLECTION)]))
        assert s.total == 0 and s.collection_id == str(COLLECTION)

    async def test_pagination_does_not_change_the_answer(self):
        a = await collect_stats(world(), COLLECTION, NEW, page_size=1)
        b = await collect_stats(world(), COLLECTION, NEW, page_size=100)
        assert a.to_port() == b.to_port()


class TestDocumentsBySource:
    async def test_distinct_documents_per_source_not_chunks_or_entities(self):
        store = FakeChunkStore(
            [
                chunk(0, doc="a", source="sharepoint"),
                chunk(1, doc="a", source="sharepoint"),
                chunk(0, doc="b", source="sharepoint"),
                chunk(0, doc="c", source="google_drive"),
                chunk(0, doc="d"),
                chunk(0, doc="z", source="sharepoint", collection=OTHER_COLLECTION),
            ]
        )
        s = await stats(store=store)
        assert s.documents == 4
        assert s.to_port()["documentsBySource"] == [
            {"source": "sharepoint", "documents": 2},
            {"source": "google_drive", "documents": 1},
            {"source": "unknown", "documents": 1},
        ]

    async def test_a_document_in_two_sources_counts_in_each_but_once_in_total(self):
        store = FakeChunkStore(
            [chunk(0, doc="a", source="sharepoint"), chunk(1, doc="a", source="google_drive")]
        )
        s = await stats(store=store)
        assert s.documents == 1
        assert s.documents_by_source == {"sharepoint": 1, "google_drive": 1}

    async def test_an_empty_collection_has_none(self):
        s = await stats(store=FakeChunkStore([]))
        assert (s.documents, s.documents_by_source) == (0, {})


class TestEmptyTarget:
    async def test_callable_with_an_empty_target(self):
        s = await stats(target="")
        assert s.other_version == s.labellable == 8  # nothing is at a version of ""
        assert s.documents_at_version == 0
        assert s.not_labelled_chunks == 2 and s.not_labelled_documents == 1

    async def test_bootstrap_collection_with_no_prior_version(self):
        store = FakeChunkStore([chunk(i, doc=f"d{i // 2}") for i in range(6)])
        s = await stats(target="", store=store)
        assert (s.total, s.not_labelled_chunks, s.not_labelled_documents) == (6, 6, 3)
        assert s.labelled == 0 and s.error_chunks == 0 and s.previous_version("") is None


class TestNotLabelledIsSubstantiveOnly:
    async def test_a_chunk_with_an_empty_own_range_is_not_counted(self):
        empty = chunk(0, doc="d1")
        empty[2]["chunk_own_start"] = empty[2]["chunk_char_end"]  # only overlaps its neighbour
        store = FakeChunkStore(
            [
                empty,
                chunk(1, doc="d1"),
                chunk(0, doc="d2", positions=False),
                chunk(0, doc="d3", schema="code_file_entity"),
            ]
        )
        s = await stats(target="", store=store)
        assert (s.not_labelled_chunks, s.not_labelled_documents) == (1, 1)

    async def test_labelled_chunks_are_not_counted(self):
        store = FakeChunkStore([chunk(0, status="ASSIGNED", version=OLD)])
        assert (await stats(store=store)).not_labelled_chunks == 0


class TestBarrier:
    async def test_other_version_reaches_zero_when_every_labellable_chunk_is_at_the_target(self):
        store = FakeChunkStore(
            [
                chunk(0, status="ASSIGNED", version=NEW),
                chunk(0, doc="c", schema="code_file_entity"),  # never labelled, must not block
                chunk(0, doc="u", positions=False),
            ]
        )
        s = await stats(store=store)
        assert s.other_version == 0 and s.total == 3

    async def test_a_chunk_at_another_version_blocks_it(self):
        store = FakeChunkStore([chunk(0, status="ASSIGNED", version=OLD)])
        assert (await stats(store=store)).other_version == 1


class TestPreviousVersion:
    async def test_most_common_other_version(self):
        store = FakeChunkStore(
            [
                chunk(0, status="ASSIGNED", version=OLD),
                chunk(1, status="ASSIGNED", version=OLD),
                chunk(2, status="ASSIGNED", version="acme-old"),
                chunk(3, status="ASSIGNED", version=NEW),
                chunk(4),
            ]
        )
        s = await stats(store=store)
        assert s.previous_version(NEW) == OLD

    async def test_none_when_everything_is_at_the_target(self):
        s = await stats(store=FakeChunkStore([chunk(0, status="ASSIGNED", version=NEW)]))
        assert s.previous_version(NEW) is None


class TestPieces:
    async def test_substantive_pieces_exclude_administrative_ones(self):
        c = chunk(0, status="ASSIGNED", version=NEW, pieces=5, unresolved=2)
        c[2]["taxonomy_admin_pieces"] = 1
        s = await stats(store=FakeChunkStore([c]))
        assert (s.pieces, s.unresolved_pieces) == (4, 2)


def test_collection_id_is_a_uuid_string():
    assert UUID(str(COLLECTION))
