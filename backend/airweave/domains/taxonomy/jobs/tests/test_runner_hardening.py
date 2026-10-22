"""The jobs runner under the review's cases.

Claim first, cancel mid-scan and mid-upload, one lease for the whole job, one build per job, an
unknown bill, and the schema in a proposal's chunk ids.
"""

import asyncio
import json

import pytest

from airweave.domains.taxonomy.jobs.candidates import CLAIM, REPORT, CandidateWriter
from airweave.domains.taxonomy.jobs.gateway import CostUnknown
from airweave.domains.taxonomy.jobs.records import (
    CANCELLED,
    COMPLETED,
    FAILED,
    KIND_PROPOSE,
    InMemoryTaxonomyJobRepository,
)
from airweave.domains.taxonomy.jobs.runner import TaxonomyJobRunner
from airweave.domains.taxonomy.jobs.tests.helpers import (
    CANDIDATE,
    OLD,
    TENANT,
    FakeChunkStore,
    FakeGateway,
    FakeResolver,
    FakeS3,
    chunk,
    docs,
    job,
)

FOLDER = f"taxonomy/tenants/{TENANT}/candidates/{CANDIDATE}/"


def make(tmp_path, store=None, gateway=None, s3=None, repo=None, owner="w1", **kw):
    store = store or FakeChunkStore(docs())
    s3 = s3 or FakeS3()
    gateway = gateway or FakeGateway()
    repo = repo or InMemoryTaxonomyJobRepository()  # real clock: leases really expire
    kw.setdefault("min_documents", 3)
    runner = TaxonomyJobRunner(
        repo,
        lambda: store,
        CandidateWriter(s3, "taxonomy/"),
        gateway,
        FakeResolver(),
        work_dir=tmp_path / "jobs",
        owner=owner,
        **kw,
    )
    return runner, repo, store, s3, gateway


class TestClaimBeforeTheScan:
    async def test_a_version_another_job_owns_fails_before_vespa_is_read(self, tmp_path):
        runner, repo, store, s3, gw = make(tmp_path)
        s3.objects[FOLDER + CLAIM] = json.dumps(
            {"job_id": "someone-else", "kind": "discover"}
        ).encode()
        j = job()
        await repo.create(j)

        await runner.run_job(j.id)

        done = await repo.get(j.id)
        assert done.state == FAILED and "already claimed" in done.error
        assert store.scrolls == 0 and gw.calls == []
        assert FOLDER + CLAIM in s3.objects  # the other job's claim is untouched

    async def test_a_claim_is_given_up_when_there_was_nothing_to_build(self, tmp_path):
        runner, repo, _, s3, _ = make(tmp_path, store=FakeChunkStore(docs(n=2)), min_documents=12)
        j = job()
        await repo.create(j)

        await runner.run_job(j.id)

        assert (await repo.get(j.id)).state == FAILED
        assert s3.objects == {}, "the claim must not block the version name for ever"

    async def test_another_jobs_claim_is_never_released(self, tmp_path):
        runner, repo, _, s3, _ = make(tmp_path, store=FakeChunkStore(docs(n=2)), min_documents=12)
        s3.objects[FOLDER + CLAIM] = json.dumps(
            {"job_id": "someone-else", "kind": "discover"}
        ).encode()
        j = job()
        await repo.create(j)

        await runner.run_job(j.id)

        assert FOLDER + CLAIM in s3.objects


class TestCancelDuringTheWork:
    async def test_cancel_seen_after_a_page_of_the_scan_stops_it_and_reads_nothing_more(
        self, tmp_path
    ):
        store = FakeChunkStore(docs())
        runner, repo, _, s3, gw = make(tmp_path, store=store)
        j = job()
        await repo.create(j)
        real = store.scroll

        async def cancelling(*a, **k):
            result = await real(*a, **k)
            await repo.request_cancel(j.id)  # arrives while the first scroll is in flight
            return result

        store.scroll = cancelling

        await runner.run_job(j.id)

        assert (await repo.get(j.id)).state == CANCELLED
        assert store.scrolls == 1 and gw.calls == []
        assert s3.objects == {}, "a cancelled scan gives its claim up too"

    async def test_cancel_during_the_upload_writes_no_report(self, tmp_path):
        runner, repo, _, s3, _ = make(tmp_path)
        j = job()
        await repo.create(j)
        real = s3.write_file_once
        written = []

        async def cancelling(path, content):
            result = await real(path, content)
            written.append(path)
            if path.endswith("dense_prototypes.npy"):  # the first file of the candidate
                await repo.request_cancel(j.id)
            return result

        s3.write_file_once = cancelling

        await runner.run_job(j.id)

        assert (await repo.get(j.id)).state == CANCELLED
        assert FOLDER + REPORT not in s3.objects, "no report once the cancel was seen"
        assert FOLDER + "taxonomy.yaml" not in s3.objects  # nothing after the cancel either


class TestLeaseAcrossTheWholeJob:
    async def test_a_slow_scan_does_not_let_the_lease_lapse(self, tmp_path):
        store = FakeChunkStore(docs())
        runner, repo, _, _, gw = make(tmp_path, store=store, lease_seconds=3)
        rival, *_ = make(tmp_path, store=store, repo=repo, owner="w2", lease_seconds=3)
        j = job()
        await repo.create(j)
        real = store.scroll
        outcome = {}

        async def slow(*a, **k):
            await asyncio.sleep(4.2)  # longer than the lease: only the heartbeat keeps it
            outcome["rival"] = await rival.run_job(j.id)
            return await real(*a, **k)

        store.scroll = slow

        await runner.run_job(j.id)

        assert outcome["rival"] == "busy", "another worker claimed the job mid-scan"
        assert (await repo.get(j.id)).state == COMPLETED and len(gw.calls) == 1


class TestOneBuildPerJob:
    async def test_a_retry_waits_for_the_cancelled_attempts_thread_instead_of_building_twice(
        self, tmp_path
    ):
        runner, repo, _, _, gw = make(tmp_path)
        gw.delay = 1.0
        j = job()
        await repo.create(j)
        first = asyncio.create_task(runner.run_job(j.id))
        await asyncio.sleep(0.4)  # the build is running on its thread
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        await runner.run_job(j.id)  # the next attempt

        assert len(gw.calls) == 1, "a second build started beside the first one"
        assert (await repo.get(j.id)).state == COMPLETED
        assert not runner._builds

    async def test_a_result_finished_just_as_the_attempt_is_cancelled_is_kept_for_the_retry(
        self, tmp_path, monkeypatch
    ):
        from airweave.domains.taxonomy.jobs import runner as runner_module

        runner, repo, _, _, gw = make(tmp_path)
        j = job()
        await repo.create(j)
        real_shield = asyncio.shield

        async def cancelled_after_the_thread_finished(future):
            await real_shield(future)
            raise asyncio.CancelledError  # the result is ready, but the attempt never reads it

        monkeypatch.setattr(runner_module.asyncio, "shield", cancelled_after_the_thread_finished)
        with pytest.raises(asyncio.CancelledError):
            await runner.run_job(j.id)
        monkeypatch.undo()

        await runner.run_job(j.id)  # the retry

        assert len(gw.calls) == 1, "the paid build ran again"
        assert (await repo.get(j.id)).state == COMPLETED


class TestCostUnknown:
    async def test_a_build_whose_ledger_cannot_be_read_fails_instead_of_costing_zero(
        self, tmp_path
    ):
        gw = FakeGateway()
        gw.error = CostUnknown("cost ledger cost_ledger.json is missing or unreadable")
        runner, repo, _, s3, _ = make(tmp_path, gateway=gw)
        j = job()
        await repo.create(j)

        await runner.run_job(j.id)

        done = await repo.get(j.id)
        assert done.state == FAILED and "cost unknown" in done.error
        assert not [p for p in s3.objects if not p.endswith(CLAIM)]


class TestProposalChunkIdentity:
    async def test_chunk_ids_keep_the_schema_and_documents_are_qualified(self, tmp_path):
        rows = [
            chunk(i, doc=f"d{i}", status="UNRESOLVED", version=OLD, unresolved=1) for i in range(4)
        ]
        # The same document id under another schema must stay a different chunk
        rows.append(
            chunk(
                0, doc="d0", schema="other_entity", status="UNRESOLVED", version=OLD, unresolved=1
            )
        )
        runner, repo, _, _, gw = make(tmp_path, store=FakeChunkStore(rows))
        j = job(kind=KIND_PROPOSE)
        await repo.create(j)

        await runner.run_job(j.id)

        chunks = gw.calls[0]["chunks"]
        ids = [c["chunk_id"] for c in chunks]
        assert len(set(ids)) == len(ids) == 5
        assert all("/" in i and i.split("/")[0] in {"file_entity", "other_entity"} for i in ids)
        docs_seen = {c["document_id"] for c in chunks}
        assert "/file_entity/d0" in docs_seen and "/other_entity/d0" in docs_seen
