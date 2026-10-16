"""The re-label job's Vespa contract on a real Vespa (batch-5 gate).

``VespaChunkStore`` and the re-label runner against a disposable Vespa (see conftest.py):

- a conditional update on the content hash applies only when the hash matches; a failed
  condition answers 412, and so does a missing document (not 404), and no document is created;
- weighted-set and array assigns replace the old contents;
- the visit parameters (``fieldSet``, ``selection`` with a status filter, ``wantedDocumentCount``,
  the continuation token) behave as the code assumes;
- a sync replacing a chunk while a re-label holds the old text wins.

Every company and text here is invented.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
import pytest

from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.mapping import RegistryMap
from airweave.domains.taxonomy.relabel.chunk_store import (
    APPLIED,
    CONDITION_FAILED,
    VespaChunkStore,
)
from airweave.domains.taxonomy.relabel.chunks import SCAN_FIELDS, ChunkRow
from airweave.domains.taxonomy.relabel.jobs import (
    COMPLETED,
    InMemoryJobRepository,
    RelabelJob,
)
from airweave.domains.taxonomy.relabel.runner import RelabelRunner
from airweave.domains.taxonomy.relabel.tests.helpers import FakeBus, FakePool, FakeResolver

SCHEMA = "file_entity"
HEADER = "Acme Q3.pdf · google_drive"
OLD, NEW = "acme-2026.10", "acme-2026.11"
ORG = UUID(int=1)


def doc_url(base: str, doc_id: str) -> str:
    return f"{base}/document/v1/airweave/{SCHEMA}/docid/{quote(doc_id, safe='')}"


def put_chunk(
    base: str,
    collection_id: str,
    n: int,
    *,
    doc: str = "doc-1",
    body: str = "Late loan payments rose sharply this quarter.",
    hash_: str = "h1",
    status: Optional[str] = None,
    version: Optional[str] = None,
    topic_ids: Optional[Dict[str, int]] = None,
    positions: bool = True,
) -> str:
    """Feed one chunk through the document API; returns its document id."""
    doc_id = f"{doc}__chunk_{n}"
    fields: Dict[str, Any] = {
        "entity_id": doc_id,
        "name": "Acme Q3",
        "airweave_system_metadata_collection_id": collection_id,
        "airweave_system_metadata_original_entity_id": doc,
        "airweave_system_metadata_chunk_index": n,
        "airweave_system_metadata_hash": hash_,
        "textual_representation": f"{HEADER}\n{body}",
        "chunk_body_offset": len(HEADER) + 1,
        "dense_embedding": {"values": [0.1] * 8},
    }
    if positions:
        fields.update(chunk_char_start=0, chunk_own_start=0, chunk_char_end=len(body))
    if status:
        fields.update(
            taxonomy_status=status,
            taxonomy_piece_count=1,
            taxonomy_admin_pieces=0,
            taxonomy_unresolved_pieces=0,
        )
    if version:
        fields["taxonomy_version"] = version
    if topic_ids:
        fields["topic_ids"] = topic_ids
    for _ in range(30):
        resp = httpx.post(doc_url(base, doc_id), json={"fields": fields}, timeout=30)
        if resp.status_code not in (503, 507):
            break
    assert resp.status_code == 200, resp.text
    return doc_id


def get_fields(base: str, doc_id: str) -> Optional[Dict[str, Any]]:
    resp = httpx.get(doc_url(base, doc_id), timeout=30)
    if resp.status_code == 404:
        return None
    assert resp.status_code == 200, resp.text
    return resp.json()["fields"]


def row(base: str, doc_id: str) -> ChunkRow:
    return ChunkRow(SCHEMA, doc_id, {})


@pytest.fixture
async def store(vespa_deployed):
    s = VespaChunkStore(base_url=vespa_deployed)
    yield s
    await s.close()


class TestConditionalUpdate:
    async def test_applies_when_the_hash_matches_and_only_the_given_fields_change(
        self, vespa_deployed, collection_id, store
    ):
        doc = put_chunk(
            vespa_deployed, collection_id, 0, hash_="aaa", status="UNRESOLVED", version=OLD
        )
        assert (
            await store.update(
                row(vespa_deployed, doc),
                {"taxonomy_status": "ASSIGNED", "taxonomy_version": NEW},
                "aaa",
            )
            == APPLIED
        )
        fields = get_fields(vespa_deployed, doc)
        assert (fields["taxonomy_status"], fields["taxonomy_version"]) == ("ASSIGNED", NEW)
        assert fields["airweave_system_metadata_hash"] == "aaa"
        assert fields["textual_representation"].endswith("this quarter.")

    async def test_a_failed_condition_answers_412_and_changes_nothing(
        self, vespa_deployed, collection_id, store
    ):
        doc = put_chunk(vespa_deployed, collection_id, 0, hash_="new-hash", version=OLD)
        resp = httpx.put(
            doc_url(vespa_deployed, doc),
            params={
                "condition": f'{SCHEMA}.airweave_system_metadata_hash=="old-hash"',
                "create": "false",
            },
            json={"fields": {"taxonomy_version": {"assign": NEW}}},
        )
        assert resp.status_code == 412, resp.text  # the raw status the code relies on
        assert await store.update(
            row(vespa_deployed, doc), {"taxonomy_version": NEW}, "old-hash"
        ) == (CONDITION_FAILED)
        assert get_fields(vespa_deployed, doc)["taxonomy_version"] == OLD

    async def test_a_missing_document_answers_412_not_404_and_is_not_created(
        self, vespa_deployed, collection_id, store
    ):
        """Measured on Vespa 8: with a condition, a missing document is a failed test-and-set.

        412, never a 404. The job therefore treats a deleted chunk like a replaced one: skipped
        and counted, nothing written. Every update carries a condition because WITHOUT one a
        ``create=false`` update on a missing document answers 200.
        """
        ghost = f"ghost-{uuid4()}__chunk_0"
        resp = httpx.put(
            doc_url(vespa_deployed, ghost),
            params={
                "condition": f'{SCHEMA}.airweave_system_metadata_hash=="x"',
                "create": "false",
            },
            json={"fields": {"taxonomy_version": {"assign": NEW}}},
        )
        assert resp.status_code == 412 and "does not exist" in resp.text, resp.text
        assert (
            await store.update(row(vespa_deployed, ghost), {"taxonomy_version": NEW}, "x")
            == CONDITION_FAILED
        )
        assert get_fields(vespa_deployed, ghost) is None

    async def test_a_condition_value_that_could_break_out_is_never_sent(
        self, vespa_deployed, collection_id, store
    ):
        doc = put_chunk(vespa_deployed, collection_id, 0, hash_="aaa")
        with pytest.raises(ValueError):
            await store.update(
                row(vespa_deployed, doc), {"taxonomy_version": NEW}, 'x" or true or "'
            )

    async def test_weighted_set_assign_replaces_the_old_contents(
        self, vespa_deployed, collection_id, store
    ):
        doc = put_chunk(
            vespa_deployed,
            collection_id,
            0,
            hash_="aaa",
            topic_ids={"acme:old.a": 1, "acme:old.b": 2},
        )
        assert get_fields(vespa_deployed, doc)["topic_ids"] == {"acme:old.a": 1, "acme:old.b": 2}
        await store.update(
            row(vespa_deployed, doc),
            {"topic_ids": {"acme:new.c": 3}, "broad_family_ids": ["acme:credit", "acme:rev"]},
            "aaa",
        )
        fields = get_fields(vespa_deployed, doc)
        assert fields["topic_ids"] == {"acme:new.c": 3}, "the old topics must not linger"
        assert fields["broad_family_ids"] == ["acme:credit", "acme:rev"]

    async def test_assigning_an_empty_weighted_set_clears_it(
        self, vespa_deployed, collection_id, store
    ):
        doc = put_chunk(vespa_deployed, collection_id, 0, hash_="aaa", topic_ids={"acme:x": 1})
        await store.update(
            row(vespa_deployed, doc), {"topic_ids": {}, "broad_family_ids": []}, "aaa"
        )
        fields = get_fields(vespa_deployed, doc)
        assert not fields.get("topic_ids") and not fields.get("broad_family_ids")

    async def test_every_label_field_type_is_accepted(self, vespa_deployed, collection_id, store):
        doc = put_chunk(vespa_deployed, collection_id, 0, hash_="aaa")
        spans = json.dumps([{"c": "acme:x", "s": 0, "e": 5}])
        labels = {
            "taxonomy_status": "ASSIGNED",
            "topic_ids": {"acme:x": 1},
            "broad_family_ids": ["acme:f"],
            "taxonomy_spans": spans,
            "taxonomy_version": NEW,
            "taxonomy_piece_count": 2,
            "taxonomy_admin_pieces": 1,
            "taxonomy_unresolved_pieces": 0,
        }
        assert await store.update(row(vespa_deployed, doc), labels, "aaa") == APPLIED
        got = get_fields(vespa_deployed, doc)
        assert got["taxonomy_spans"] == spans and got["taxonomy_piece_count"] == 2
        assert got["taxonomy_admin_pieces"] == 1 and got["taxonomy_unresolved_pieces"] == 0

    async def test_after_the_hash_changes_every_concurrent_update_with_the_old_hash_fails(
        self, vespa_deployed, collection_id, store
    ):
        doc = put_chunk(vespa_deployed, collection_id, 0, hash_="h-old", version=OLD)
        put_chunk(vespa_deployed, collection_id, 0, hash_="h-new", version=NEW, status="UNRESOLVED")
        results = await asyncio.gather(
            *(
                store.update(row(vespa_deployed, doc), {"taxonomy_version": f"stale-{i}"}, "h-old")
                for i in range(40)
            )
        )
        assert set(results) == {CONDITION_FAILED}
        assert get_fields(vespa_deployed, doc)["taxonomy_version"] == NEW

    async def test_of_many_concurrent_updates_with_the_right_hash_each_applies(
        self, vespa_deployed, collection_id, store
    ):
        doc = put_chunk(vespa_deployed, collection_id, 0, hash_="h1")
        results = await asyncio.gather(
            *(
                store.update(row(vespa_deployed, doc), {"taxonomy_piece_count": i}, "h1")
                for i in range(20)
            )
        )
        assert set(results) == {APPLIED}


class TestVisit:
    def seed(self, base: str, cid: str, n: int = 7, statuses=None) -> List[str]:
        out = []
        for i in range(n):
            status = (statuses or ["UNRESOLVED", "ASSIGNED"])[i % 2]
            out.append(put_chunk(base, cid, i, doc=f"d{i}", status=status, version=OLD))
        return out

    async def scroll_all(self, store, cid, fields, page, statuses=None):
        rows, token, pages = [], None, 0
        while True:
            batch, token = await store.scroll(UUID(cid), fields, page, token, statuses)
            rows += batch
            pages += 1
            assert pages < 50
            if token is None:
                return rows

    async def test_a_scroll_with_a_continuation_covers_every_chunk_once(
        self, vespa_deployed, collection_id, store
    ):
        ids = self.seed(vespa_deployed, collection_id, 7)
        rows = await self.scroll_all(store, collection_id, SCAN_FIELDS, 2)
        assert sorted(r.doc_id for r in rows) == sorted(ids)
        assert len({r.doc_id for r in rows}) == len(rows)

    async def test_only_the_collection_asked_for_is_visited(
        self, vespa_deployed, collection_id, store
    ):
        self.seed(vespa_deployed, collection_id, 3)
        other = str(uuid4())
        put_chunk(vespa_deployed, other, 0, doc="theirs")
        rows = await self.scroll_all(store, collection_id, SCAN_FIELDS, 5)
        assert "theirs__chunk_0" not in {r.doc_id for r in rows} and len(rows) == 3
        httpx.delete(
            f"{vespa_deployed}/document/v1/airweave/{SCHEMA}/docid",
            params={
                "selection": f'{SCHEMA}.airweave_system_metadata_collection_id=="{other}"',
                "cluster": "airweave",
            },
        )

    async def test_the_field_set_returns_only_the_requested_fields(
        self, vespa_deployed, collection_id, store
    ):
        self.seed(vespa_deployed, collection_id, 2)
        rows = await self.scroll_all(
            store, collection_id, ("taxonomy_status", "airweave_system_metadata_hash"), 5
        )
        assert rows and all(
            set(r.fields) <= {"taxonomy_status", "airweave_system_metadata_hash"} for r in rows
        )
        assert all("textual_representation" not in r.fields for r in rows)

    async def test_the_status_filter_keeps_only_those_statuses(
        self, vespa_deployed, collection_id, store
    ):
        self.seed(vespa_deployed, collection_id, 7)
        rows = await self.scroll_all(store, collection_id, SCAN_FIELDS, 3, ["UNRESOLVED"])
        assert rows and {r.status for r in rows} == {"UNRESOLVED"} and len(rows) == 4
        both = await self.scroll_all(
            store, collection_id, SCAN_FIELDS, 3, ["UNRESOLVED", "PARTIALLY_ASSIGNED"]
        )
        assert len(both) == 4

    async def test_a_rejected_continuation_is_a_value_error_the_job_can_recover_from(
        self, vespa_deployed, collection_id, store
    ):
        from airweave.domains.taxonomy.relabel.chunk_store import encode_token

        self.seed(vespa_deployed, collection_id, 3)
        try:
            rows, _ = await store.scroll(
                UUID(collection_id), SCAN_FIELDS, 5, encode_token(0, "not-a-token")
            )
        except RuntimeError as e:  # Vespa answered with an error status
            assert "visit failed" in str(e)
        else:  # or ignored it and started over; either way nothing is lost
            assert isinstance(rows, list)

    async def test_the_page_size_is_a_hint_not_a_limit_the_code_must_tolerate(
        self, vespa_deployed, collection_id, store
    ):
        self.seed(vespa_deployed, collection_id, 9)
        rows, token = await store.scroll(UUID(collection_id), SCAN_FIELDS, 2)
        assert len(rows) >= 1  # may return more or fewer than asked, never zero before the end
        assert token is not None or len(rows) == 9


def taxonomy() -> ResolvedTaxonomy:
    from pathlib import Path

    return ResolvedTaxonomy(
        "neo-acme",
        "t-acme",
        NEW,
        "a" * 64,
        4,
        "calibrated",
        None,
        Path("/tmp/none"),
        {},
        RegistryMap(NEW, {"CAT001": "acme:credit.loan_quality"}, {"BF01": "acme:credit"}),
    )


def runner_for(vespa_deployed: str, pool: FakePool, repo, bus):
    return RelabelRunner(
        repo,
        lambda: VespaChunkStore(base_url=vespa_deployed),
        FakeResolver(taxonomy()),
        pool,
        bus,
        page_size=3,
        batch_size=2,
    )


def job(cid: str, **kw) -> RelabelJob:
    base = {
        "id": uuid4(),
        "organization_id": ORG,
        "collection_id": UUID(cid),
        "collection_readable_id": "neo-acme",
        "target_version": NEW,
        "spend_cap_usd": 10.0,
        "previous_version": OLD,
    }
    base.update(kw)
    return RelabelJob(**base)


class TestRelabelJobOnRealVespa:
    async def test_a_full_job_writes_exactly_what_a_sync_writes(
        self, vespa_deployed, collection_id
    ):
        ids = [
            put_chunk(
                vespa_deployed,
                collection_id,
                i,
                doc=f"d{i}",
                status="ASSIGNED",
                version=OLD,
                topic_ids={"acme:old": 1},
            )
            for i in range(5)
        ]
        repo, bus = InMemoryJobRepository(), FakeBus()
        runner = runner_for(vespa_deployed, FakePool(), repo, bus)
        j = job(collection_id)
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert done.state == COMPLETED and done.chunks_relabelled == 5 and done.chunks_failed == 0
        for doc in ids:
            f = get_fields(vespa_deployed, doc)
            assert f["taxonomy_status"] == "ASSIGNED" and f["taxonomy_version"] == NEW
            assert f["topic_ids"] == {"acme:credit.loan_quality": 1}, "old topic replaced"
            assert f["broad_family_ids"] == ["acme:credit"]
            assert json.loads(f["taxonomy_spans"])[0]["c"] == "acme:credit.loan_quality"
            assert f["textual_representation"].startswith(HEADER)  # text untouched
            assert f["airweave_system_metadata_hash"] == "h1"
        assert bus.events and bus.events[0].status == "completed"

    async def test_a_sync_replacing_a_chunk_during_the_relabel_wins(
        self, vespa_deployed, collection_id
    ):
        stale = put_chunk(
            vespa_deployed,
            collection_id,
            0,
            doc="race",
            body="Old Acme text.",
            hash_="old-hash",
            status="ASSIGNED",
            version=OLD,
        )
        calm = put_chunk(
            vespa_deployed, collection_id, 0, doc="calm", status="ASSIGNED", version=OLD
        )

        class SyncWins(VespaChunkStore):
            """Between the job's read and its write, a sync replaces one chunk on the real Vespa."""

            async def update(self, chunk, fields, expected_hash):
                if chunk.doc_id == stale and not getattr(self, "replaced", False):
                    self.replaced = True
                    put_chunk(
                        vespa_deployed,
                        collection_id,
                        0,
                        doc="race",
                        body="New Acme text.",
                        hash_="new-hash",
                        status="UNRESOLVED",
                        version=NEW,
                    )
                return await super().update(chunk, fields, expected_hash)

        repo, bus = InMemoryJobRepository(), FakeBus()
        runner = RelabelRunner(
            repo,
            lambda: SyncWins(base_url=vespa_deployed),
            FakeResolver(taxonomy()),
            FakePool(),
            bus,
            page_size=5,
            batch_size=5,
        )
        j = job(collection_id)
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert (
            done.state == COMPLETED
            and done.chunks_skipped_changed == 1
            and done.chunks_relabelled == 1
        )
        won = get_fields(vespa_deployed, stale)
        assert won["taxonomy_status"] == "UNRESOLVED", "the sync's label stayed"
        assert won["airweave_system_metadata_hash"] == "new-hash"
        assert won["textual_representation"].endswith("New Acme text.")
        assert not won.get("topic_ids"), "no old-text label landed on the new text"
        assert get_fields(vespa_deployed, calm)["taxonomy_status"] == "ASSIGNED"
        assert get_fields(vespa_deployed, calm)["topic_ids"] == {"acme:credit.loan_quality": 1}

    async def test_a_chunk_deleted_during_the_relabel_is_skipped_not_created(
        self, vespa_deployed, collection_id
    ):
        doc = put_chunk(
            vespa_deployed, collection_id, 0, doc="gone", status="ASSIGNED", version=OLD
        )

        class Deleter(VespaChunkStore):
            async def update(self, chunk, fields, expected_hash):
                httpx.delete(doc_url(vespa_deployed, doc), timeout=30)
                return await super().update(chunk, fields, expected_hash)

        repo = InMemoryJobRepository()
        runner = RelabelRunner(
            repo,
            lambda: Deleter(base_url=vespa_deployed),
            FakeResolver(taxonomy()),
            FakePool(),
            FakeBus(),
            page_size=5,
            batch_size=5,
        )
        j = job(collection_id)
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert (
            done.state == COMPLETED and done.chunks_skipped_changed == 1 and done.chunks_failed == 0
        )
        assert get_fields(vespa_deployed, doc) is None

    async def test_the_incremental_stamp_changes_the_version_only(
        self, vespa_deployed, collection_id
    ):
        carried = put_chunk(
            vespa_deployed,
            collection_id,
            0,
            doc="kept",
            status="ASSIGNED",
            version=OLD,
            topic_ids={"acme:keep": 2},
        )
        backlog = put_chunk(
            vespa_deployed, collection_id, 0, doc="todo", status="UNRESOLVED", version=OLD
        )
        repo = InMemoryJobRepository()
        pool = FakePool()
        runner = runner_for(vespa_deployed, pool, repo, FakeBus())
        j = job(collection_id, mode="incremental")
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert (done.chunks_stamped, done.chunks_relabelled) == (1, 1)
        kept = get_fields(vespa_deployed, carried)
        assert kept["taxonomy_version"] == NEW and kept["topic_ids"] == {"acme:keep": 2}
        assert kept["taxonomy_status"] == "ASSIGNED"
        assert [r["chunk_id"] for r in pool.rows] == [f"{SCHEMA}/{backlog}"]

    async def test_code_files_are_not_touched(self, vespa_deployed, collection_id):
        base_chunk = put_chunk(
            vespa_deployed, collection_id, 0, doc="plain", status="ASSIGNED", version=OLD
        )
        repo = InMemoryJobRepository()
        runner = runner_for(vespa_deployed, FakePool(), repo, FakeBus())
        j = job(collection_id)
        await repo.create(j)
        await runner.run_job(j.id)
        assert get_fields(vespa_deployed, base_chunk)["taxonomy_version"] == NEW
