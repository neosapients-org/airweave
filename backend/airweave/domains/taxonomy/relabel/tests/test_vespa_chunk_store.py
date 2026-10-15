"""``VespaChunkStore``'s HTTP calls, checked against a mock transport.

This checks what we send and how we read the answers. **It is not a Vespa**: the document
selection, ``fieldSet``, ``condition`` and ``create=false`` semantics are unverified until the
manual Vespa gate runs them on a real instance.
"""

import json
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import pytest

from airweave.domains.taxonomy.relabel.chunk_store import (
    APPLIED,
    CONDITION_FAILED,
    NOT_FOUND,
    VespaChunkStore,
    decode_token,
    encode_token,
)
from airweave.domains.taxonomy.relabel.chunks import ChunkRow
from airweave.domains.taxonomy.relabel.tests.helpers import COLLECTION

ROW = ChunkRow("file_entity", "file_entity_doc-1__chunk_0", {})


def store_with(handler):
    store = VespaChunkStore(base_url="http://vespa:8080")
    store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return store


class TestUpdate:
    async def test_put_with_the_hash_condition_and_assign_syntax(self):
        seen = {}

        def handler(request):
            seen["method"], seen["url"], seen["body"] = (
                request.method,
                request.url,
                json.loads(request.content),
            )
            return httpx.Response(200, json={})

        outcome = await store_with(handler).update(
            ROW, {"taxonomy_status": "ASSIGNED", "topic_ids": {"acme:x": 1}}, "abc123"
        )
        assert outcome == APPLIED and seen["method"] == "PUT"
        url = seen["url"]
        assert url.path == "/document/v1/airweave/file_entity/docid/file_entity_doc-1__chunk_0"
        query = parse_qs(url.query.decode())
        assert query["condition"] == ['file_entity.airweave_system_metadata_hash=="abc123"']
        assert query["create"] == ["false"]
        assert seen["body"] == {
            "fields": {
                "taxonomy_status": {"assign": "ASSIGNED"},
                "topic_ids": {"assign": {"acme:x": 1}},
            }
        }

    @pytest.mark.parametrize("status, expected", [(412, CONDITION_FAILED), (404, NOT_FOUND)])
    async def test_condition_failed_and_missing_are_outcomes_not_errors(self, status, expected):
        assert (
            await store_with(lambda r: httpx.Response(status, text="x")).update(ROW, {"a": 1}, "h")
            == expected
        )

    async def test_other_failures_raise(self):
        with pytest.raises(RuntimeError, match="500"):
            await store_with(lambda r: httpx.Response(500, text="boom")).update(ROW, {"a": 1}, "h")

    @pytest.mark.parametrize("bad", ['x"; true; "', "a b", "", "h" * 300, 'a"b'])
    async def test_a_hash_that_could_break_the_condition_is_refused(self, bad):
        called = []
        store = store_with(lambda r: called.append(r) or httpx.Response(200))
        with pytest.raises(ValueError):
            await store.update(ROW, {"a": 1}, bad)
        assert called == []


class TestScroll:
    def pages(self):
        docs = {
            "base_entity": [
                [],
            ],
            "file_entity": [[{"id": "id:airweave:file_entity::f_1", "fields": {"a": 1}}], None],
        }

        def handler(request):
            schema = request.url.path.split("/")[4]
            query = parse_qs(request.url.query.decode())
            assert query["wantedDocumentCount"] == ["2"]
            assert "airweave_system_metadata_collection_id" in query["selection"][0]
            assert str(COLLECTION) in query["selection"][0]
            assert query["fieldSet"][0].startswith(f"{schema}:") and "a,b" in query["fieldSet"][0]
            if schema == "file_entity" and "continuation" not in query:
                return httpx.Response(
                    200, json={"documents": docs[schema][0], "continuation": "tok"}
                )
            if schema == "file_entity":
                return httpx.Response(
                    200,
                    json={
                        "documents": [{"id": "id:airweave:file_entity::f_2", "fields": {"a": 2}}]
                    },
                )
            return httpx.Response(200, json={"documents": []})

        return handler

    async def test_walks_every_schema_and_resumes_with_the_token(self):
        store = store_with(self.pages())
        seen, token = [], None
        for _ in range(10):
            rows, token = await store.scroll(COLLECTION, ["a", "b"], 2, token)
            seen += [(r.schema, r.doc_id) for r in rows]
            if token is None:
                break
        assert seen == [("file_entity", "f_1"), ("file_entity", "f_2")]

    async def test_status_filter_goes_into_the_selection(self):
        captured = []

        def handler(request):
            captured.append(unquote(urlparse(str(request.url)).query))
            return httpx.Response(200, json={"documents": []})

        await store_with(handler).scroll(
            COLLECTION, ["a"], 5, None, ["UNRESOLVED", "PARTIALLY_ASSIGNED"]
        )
        assert 'taxonomy_status=="UNRESOLVED" or' in captured[0]
        assert 'taxonomy_status=="PARTIALLY_ASSIGNED"' in captured[0]

    async def test_visit_errors_raise(self):
        with pytest.raises(RuntimeError, match="visit failed"):
            await store_with(lambda r: httpx.Response(503, text="no")).scroll(COLLECTION, ["a"], 2)

    async def test_ends_when_every_schema_is_empty(self):
        rows, token = await store_with(
            lambda r: httpx.Response(200, json={"documents": []})
        ).scroll(COLLECTION, ["a"], 2)
        assert rows == [] and token is None


def test_tokens_round_trip_and_garbage_is_refused():
    assert decode_token(encode_token(3, "abc")) == (3, "abc")
    assert decode_token(None) == (0, None)
    for bad in ("nope", "[1]", '["a", null]', "[-1, null]", "[1, 2]"):
        with pytest.raises(ValueError):
            decode_token(bad)
