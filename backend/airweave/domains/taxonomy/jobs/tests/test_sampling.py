"""Sampling a collection's text for a job: document identity, a bound on memory, page hooks."""

import pytest

from airweave.domains.taxonomy.jobs.sampling import TooFewDocuments, sample_documents
from airweave.domains.taxonomy.relabel.chunks import F_SYNC
from airweave.domains.taxonomy.relabel.tests.helpers import COLLECTION, FakeChunkStore, chunk


def row(n, doc, sync="sync-a", schema="file_entity", body=None):
    # The Vespa document id is unique per sync; the source's own entity id ("doc") is not.
    kind, doc_id, fields = chunk(
        n, doc=f"{doc}@{sync}", schema=schema, **({"body": body} if body else {})
    )
    fields[F_SYNC] = sync
    fields["airweave_system_metadata_original_entity_id"] = doc
    return kind, doc_id, fields


def many(docs, chunks_each=2, **kw):
    return [row(c, d, **kw) for d in docs for c in range(chunks_each)]


class TestDocumentIdentity:
    async def test_equal_source_ids_from_two_syncs_are_two_documents(self):
        rows = [*many(["123"], sync="sync-a"), *many(["123"], sync="sync-b")]
        store = FakeChunkStore(rows)

        sample = await sample_documents(store, COLLECTION, 10, 100, min_documents=2)

        assert {c["document_id"] for c in sample} == {
            "sync-a/file_entity/123",
            "sync-b/file_entity/123",
        }
        assert len(sample) == 4  # nothing merged, nothing dropped

    async def test_equal_ids_under_two_schemas_are_two_documents(self):
        rows = [*many(["9"], schema="file_entity"), *many(["9"], schema="mail_entity")]
        store = FakeChunkStore(rows)

        sample = await sample_documents(store, COLLECTION, 10, 100, min_documents=2)

        assert len({c["document_id"] for c in sample}) == 2
        assert len({c["chunk_id"] for c in sample}) == 4

    async def test_chunk_ids_keep_the_schema(self):
        store = FakeChunkStore(many(["a", "b", "c"]))
        sample = await sample_documents(store, COLLECTION, 10, 100, min_documents=1)
        assert all(c["chunk_id"].startswith("file_entity/file_entity_") for c in sample)

    async def test_the_minimum_counts_qualified_documents(self):
        rows = [*many(["123"], sync="sync-a"), *many(["123"], sync="sync-b")]
        with pytest.raises(TooFewDocuments, match="1 substantive|at least 3"):
            await sample_documents(FakeChunkStore(rows), COLLECTION, 10, 100, min_documents=3)


class TestMemoryIsBoundedWhileReading:
    async def test_the_sample_never_exceeds_max_chunks(self):
        store = FakeChunkStore(many([f"d{i:02d}" for i in range(20)], chunks_each=3))
        sample = await sample_documents(store, COLLECTION, 20, 10, min_documents=1)
        assert 0 < len(sample) <= 10
        # whole documents only
        by_doc = {}
        for c in sample:
            by_doc.setdefault(c["document_id"], []).append(c)
        assert all(len(v) == 3 for v in by_doc.values())

    async def test_a_document_larger_than_the_bound_is_left_out_not_taken_whole(self):
        rows = [*many(["huge"], chunks_each=50), *many(["a", "b", "c"], chunks_each=2)]
        sample = await sample_documents(FakeChunkStore(rows), COLLECTION, 10, 10, min_documents=1)

        assert all("huge" not in c["document_id"] for c in sample)
        assert len(sample) == 6

    async def test_text_is_read_for_the_planned_documents_only(self):
        rows = many([f"d{i:02d}" for i in range(30)], chunks_each=2)
        store = FakeChunkStore(rows)
        reads = []
        real = store.scroll

        async def spy(collection, fields, *a, **k):
            reads.append("textual_representation" in fields)
            return await real(collection, fields, *a, **k)

        store.scroll = spy

        sample = await sample_documents(store, COLLECTION, 3, 1000, min_documents=1)

        assert len({c["document_id"] for c in sample}) == 3 and len(sample) == 6
        assert reads == [False, True]  # one count scan without text, one read of text

    async def test_the_same_sample_every_time(self):
        rows = many([f"d{i:02d}" for i in range(15)])
        first = await sample_documents(FakeChunkStore(rows), COLLECTION, 5, 100, min_documents=1)
        second = await sample_documents(FakeChunkStore(rows), COLLECTION, 5, 100, min_documents=1)
        assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]

    async def test_a_collection_that_grew_since_the_plan_is_still_bounded(self):
        rows = many(["a", "b", "c"], chunks_each=2)
        store = FakeChunkStore(rows)
        real = store.scroll
        calls = {"n": 0}

        async def growing(collection, fields, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:  # text pass: more chunks of the planned documents appeared
                for i in range(2, 40):
                    kind, did, f = row(i, "a")
                    store.docs[(kind, did)] = f
            return await real(collection, fields, *a, **k)

        store.scroll = growing

        sample = await sample_documents(store, COLLECTION, 3, 6, min_documents=1)

        assert len(sample) <= 6


class TestPageHook:
    async def test_the_hook_runs_after_every_page_of_both_scans(self):
        rows = many([f"d{i:02d}" for i in range(10)], chunks_each=2)
        store = FakeChunkStore(rows)
        seen = []

        async def on_page():
            seen.append(store.scrolls)

        import airweave.domains.taxonomy.jobs.sampling as sampling

        original = sampling.PAGE
        sampling.PAGE = 7  # several pages per scan
        try:
            await sample_documents(store, COLLECTION, 10, 100, min_documents=1, on_page=on_page)
        finally:
            sampling.PAGE = original

        assert len(seen) == store.scrolls > 2

    async def test_a_hook_that_raises_stops_the_scan_at_once(self):
        store = FakeChunkStore(many([f"d{i:02d}" for i in range(10)]))

        class Stop(Exception):
            pass

        async def on_page():
            raise Stop

        with pytest.raises(Stop):
            await sample_documents(store, COLLECTION, 10, 100, min_documents=1, on_page=on_page)
        assert store.scrolls == 1
