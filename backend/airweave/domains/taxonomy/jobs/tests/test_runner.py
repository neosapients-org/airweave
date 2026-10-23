"""The jobs runner with a fake Vespa, a fake S3 and a fake package gateway."""

import asyncio
import json
import threading

import pytest

from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError
from airweave.domains.taxonomy.jobs.candidates import REPORT, CandidateWriter
from airweave.domains.taxonomy.jobs.gateway import SpendCapReached
from airweave.domains.taxonomy.jobs.records import (
    CANCELLED,
    COMPLETED,
    FAILED,
    KIND_PROPOSE,
    RUNNING,
    InMemoryTaxonomyJobRepository,
)
from airweave.domains.taxonomy.jobs.runner import BUSY, TaxonomyJobRunner, TransientUploadError
from airweave.domains.taxonomy.jobs.tests.helpers import (
    CANDIDATE,
    NEW,
    OLD,
    TENANT,
    FakeChunkStore,
    FakeGateway,
    FakeResolver,
    FakeS3,
    chunk,
    docs,
    job,
    taxonomy,
)
from airweave.domains.taxonomy.relabel.tests.helpers import Clock

FOLDER = f"taxonomy/tenants/{TENANT}/candidates/{CANDIDATE}/"


def make(tmp_path, store=None, gateway=None, s3=None, resolver=None, repo=None, owner="w1", **kw):
    store = store or FakeChunkStore(docs())
    s3 = s3 or FakeS3()
    gateway = gateway or FakeGateway()
    repo = repo or InMemoryTaxonomyJobRepository(Clock())
    kw.setdefault("min_documents", 3)
    runner = TaxonomyJobRunner(
        repo,
        lambda: store,
        CandidateWriter(s3, "taxonomy/"),
        gateway,
        resolver or FakeResolver(),
        work_dir=tmp_path / "jobs",
        owner=owner,
        **kw,
    )
    return runner, repo, store, s3, gateway


async def run(runner, repo, **job_kw):
    j = job(**job_kw)
    await repo.create(j)
    await runner.run_job(j.id)
    return await repo.get(j.id)


class TestDiscover:
    async def test_samples_builds_and_writes_the_candidate_with_a_report(self, tmp_path):
        runner, repo, store, s3, gw = make(tmp_path)
        done = await run(runner, repo)
        assert done.state == COMPLETED and done.error is None
        assert done.result["candidate_prefix"] == FOLDER and done.spend_used_usd == 0.02
        assert sorted(done.result["files"]) == [
            "dense_prototypes.npy",
            "final_taxonomy.json",
            "taxonomy.yaml",
        ]
        assert s3.under(FOLDER) == sorted(
            FOLDER + n
            for n in [
                "_claim.json",
                REPORT,
                "dense_prototypes.npy",
                "final_taxonomy.json",
                "taxonomy.yaml",
            ]
        )
        report = json.loads(s3.objects[FOLDER + REPORT])
        assert (
            report["clusters"] and report["cost_usd"] == 0.02 and report["job_id"] == str(done.id)
        )
        assert store.updates == [] and store.closed

    async def test_only_the_candidates_prefix_is_ever_written(self, tmp_path):
        runner, repo, _, s3, _ = make(tmp_path)
        await run(runner, repo)
        assert s3.writes and all(p.startswith(FOLDER) for p in s3.writes)
        assert not [
            p for p in s3.objects if "/versions/" in p or "/active/" in p or "/models/" in p
        ]

    async def test_the_package_runs_on_a_thread_not_the_event_loop(self, tmp_path):
        runner, repo, _, _, gw = make(tmp_path)
        await run(runner, repo)
        assert gw.threads and gw.threads[0] != threading.get_ident()

    async def test_the_package_gets_the_sample_cap_and_description(self, tmp_path):
        runner, repo, _, _, gw = make(tmp_path)
        await run(
            runner,
            repo,
            spend_cap_usd=0.75,
            params={"sample_size": 5, "corpus_description": "Acme loan notes"},
        )
        call = gw.calls[0]
        assert call["spend_cap_usd"] == 0.75 and call["corpus_description"] == "Acme loan notes"
        assert call["tenant_id"] == TENANT and call["candidate_version"] == CANDIDATE
        assert len({c["document_id"] for c in call["chunks"]}) == 5

    async def test_sample_does_not_depend_on_the_candidate_version(self, tmp_path):
        a = make(tmp_path / "a")
        b = make(tmp_path / "b")
        await run(a[0], a[1], params={"sample_size": 6})
        await run(b[0], b[1], params={"sample_size": 6}, candidate_version="v-other")
        ids = lambda gw: sorted({c["document_id"] for c in gw.calls[0]["chunks"]})  # noqa: E731
        assert b[4].calls[0]["candidate_version"] == "v-other"
        assert ids(a[4]) == ids(b[4])

    async def test_sample_is_deterministic_for_a_candidate_version(self, tmp_path):
        a = make(tmp_path / "a")
        b = make(tmp_path / "b")
        await run(a[0], a[1], params={"sample_size": 6})
        await run(b[0], b[1], params={"sample_size": 6})
        ids = lambda gw: sorted({c["document_id"] for c in gw.calls[0]["chunks"]})  # noqa: E731
        assert ids(a[4]) == ids(b[4])

    async def test_chunk_text_is_the_own_text_without_the_header(self, tmp_path):
        runner, repo, _, _, gw = make(tmp_path)
        await run(runner, repo)
        texts = [c["text"] for c in gw.calls[0]["chunks"]]
        assert texts and not any("google_drive" in t for t in texts)

    async def test_code_files_and_unpositioned_chunks_are_not_sampled(self, tmp_path):
        extra = [
            chunk(0, doc="code", schema="code_file_entity"),
            chunk(0, doc="raw", positions=False),
        ]
        runner, repo, _, _, gw = make(tmp_path, store=FakeChunkStore([*docs(), *extra]))
        await run(runner, repo)
        assert not {"code", "raw"} & {c["document_id"] for c in gw.calls[0]["chunks"]}

    async def test_too_few_documents_fails_before_anything_is_claimed_or_spent(self, tmp_path):
        runner, repo, _, s3, gw = make(tmp_path, store=FakeChunkStore(docs(n=2)), min_documents=12)
        done = await run(runner, repo)
        assert done.state == FAILED and "at least 12" in done.error
        assert gw.calls == [] and s3.objects == {}

    async def test_another_collections_chunks_are_never_read(self, tmp_path):
        from airweave.domains.taxonomy.relabel.tests.helpers import OTHER_COLLECTION

        other = [chunk(i, doc="theirs", collection=OTHER_COLLECTION) for i in range(3)]
        runner, repo, _, _, gw = make(tmp_path, store=FakeChunkStore([*docs(), *other]))
        await run(runner, repo)
        assert "theirs" not in {c["document_id"] for c in gw.calls[0]["chunks"]}


class TestPropose:
    def unresolved(self):
        return [
            chunk(i, doc=f"d{i}", status="UNRESOLVED", version=OLD, unresolved=1) for i in range(6)
        ] + [chunk(0, doc="ok", status="ASSIGNED", version=OLD)]

    async def test_reads_the_unresolved_chunks_and_runs_the_wave_against_the_serving_version(
        self, tmp_path
    ):
        runner, repo, _, s3, gw = make(tmp_path, store=FakeChunkStore(self.unresolved()))
        done = await run(runner, repo, kind=KIND_PROPOSE)
        assert done.state == COMPLETED
        call = gw.calls[0]
        assert call["kind"] == "propose" and call["serving"].taxonomy_version == NEW
        assert sorted(c["document_id"] for c in call["chunks"]) == [
            f"/file_entity/d{i}" for i in range(6)
        ]
        assert s3.writes and all(p.startswith(FOLDER) for p in s3.writes)

    async def test_nothing_unresolved_completes_without_a_build_or_a_write(self, tmp_path):
        store = FakeChunkStore([chunk(0, status="ASSIGNED", version=OLD)])
        runner, repo, _, s3, gw = make(tmp_path, store=store)
        done = await run(runner, repo, kind=KIND_PROPOSE)
        assert done.state == COMPLETED and "nothing to propose" in done.result["note"]
        assert gw.calls == [] and s3.objects == {}

    @pytest.mark.parametrize(
        "resolver, message",
        [
            (FakeResolver("none"), "no active taxonomy"),
            (FakeResolver(TaxonomyResolutionError("checksum mismatch")), "checksum mismatch"),
        ],
    )
    async def test_without_a_usable_serving_version_it_fails_visibly(
        self, tmp_path, resolver, message
    ):
        runner, repo, _, s3, gw = make(
            tmp_path, store=FakeChunkStore(self.unresolved()), resolver=resolver
        )
        done = await run(runner, repo, kind=KIND_PROPOSE)
        assert (
            done.state == FAILED and message in done.error and gw.calls == [] and s3.objects == {}
        )

    async def test_a_taxonomy_of_another_tenant_is_refused(self, tmp_path):
        other = taxonomy()
        object.__setattr__(other, "tenant_id", "t-globex")
        runner, repo, _, s3, gw = make(
            tmp_path, store=FakeChunkStore(self.unresolved()), resolver=FakeResolver(other)
        )
        done = await run(runner, repo, kind=KIND_PROPOSE)
        assert done.state == FAILED and "another tenant" in done.error and s3.objects == {}

    async def test_during_a_migration_it_waits(self, tmp_path):
        busy = taxonomy()
        object.__setattr__(busy, "migration_id", "mig-1")
        runner, repo, _, _, gw = make(
            tmp_path, store=FakeChunkStore(self.unresolved()), resolver=FakeResolver(busy)
        )
        done = await run(runner, repo, kind=KIND_PROPOSE)
        assert done.state == FAILED and "migration is in progress" in done.error and gw.calls == []

    async def test_discover_does_not_need_a_serving_version(self, tmp_path):
        runner, repo, *_ = make(tmp_path, resolver=FakeResolver("none"))
        assert (await run(runner, repo)).state == COMPLETED


class TestReplayAndResume:
    async def test_a_replay_after_completion_does_not_rebuild_or_rewrite(self, tmp_path):
        runner, repo, _, s3, gw = make(tmp_path)
        done = await run(runner, repo)
        writes = list(s3.writes)
        again = job(id=done.id, idempotency_key="k1")
        repo.jobs[done.id].state = "PENDING"  # e.g. Temporal retried the activity
        await runner.run_job(done.id)
        assert len(gw.calls) == 1 and s3.writes == writes
        assert (await repo.get(done.id)).state == COMPLETED and again

    async def test_a_second_job_for_a_finished_candidate_fails_without_spending(self, tmp_path):
        runner, repo, _, s3, gw = make(tmp_path)
        await run(runner, repo)
        second = await run(runner, repo, idempotency_key="k2")
        assert second.state == FAILED and "another job" in second.error and len(gw.calls) == 1

    async def test_a_claimed_but_incomplete_folder_blocks_another_job(self, tmp_path):
        s3 = FakeS3()
        s3.objects[FOLDER + "_claim.json"] = json.dumps(
            {"job_id": "someone-else", "kind": "discover"}
        ).encode()
        runner, repo, _, _, gw = make(tmp_path, s3=s3)
        done = await run(runner, repo)
        assert done.state == FAILED and "already claimed" in done.error and gw.calls == []

    async def test_a_crash_after_the_build_resumes_without_a_second_build(self, tmp_path):
        s3 = FakeS3()
        s3.fail_on = "final_taxonomy.json"  # upload dies after the build
        clock = Clock()
        repo = InMemoryTaxonomyJobRepository(clock)
        r1, _, _, _, gw = make(tmp_path, s3=s3, repo=repo, owner="w1")
        j = job()
        await repo.create(j)
        with pytest.raises(TransientUploadError):
            await r1.run_job(j.id)  # Temporal retries the activity
        assert (await repo.get(j.id)).state == RUNNING
        s3.fail_on = None
        await r1.run_job(j.id)
        final = await repo.get(j.id)
        assert final.state == COMPLETED and len(gw.calls) == 1, (
            "the build is reused, not paid twice"
        )
        assert FOLDER + REPORT in s3.objects

    async def test_a_job_held_by_another_worker_is_left_alone(self, tmp_path):
        clock = Clock()
        repo = InMemoryTaxonomyJobRepository(clock)
        r1, *_ = make(tmp_path, repo=repo, owner="w1")
        r2, _, _, _, gw2 = (
            make(tmp_path, repo=repo, owner="w2", clock=None)
            if False
            else make(tmp_path, repo=repo, owner="w2")
        )
        j = job()
        await repo.create(j)
        await repo.claim(j.id, "w1", 300)
        assert await r2.run_job(j.id) == BUSY and gw2.calls == []
        clock.advance(301)
        assert await r2.run_job(j.id) == COMPLETED

    async def test_a_worker_that_lost_its_lease_stops_quietly(self, tmp_path):
        clock = Clock()
        repo = InMemoryTaxonomyJobRepository(clock)
        gw = FakeGateway()
        runner, *_ = make(tmp_path, repo=repo, gateway=gw, owner="w1")
        j = job()
        await repo.create(j)

        def taken():
            clock.advance(1000)
            repo.jobs[j.id].lease_owner = "w2"

        gw.on_call = taken
        await runner.run_job(j.id)
        state = await repo.get(j.id)
        assert state.state == RUNNING and state.lease_owner == "w2"

    async def test_the_working_directory_is_deleted_when_the_job_ends(self, tmp_path):
        runner, repo, *_ = make(tmp_path)
        done = await run(runner, repo)
        assert not (tmp_path / "jobs" / str(done.id)).exists()

    async def test_the_working_directory_is_kept_for_a_resume_when_the_worker_is_stopped(
        self, tmp_path
    ):
        gw = FakeGateway()
        runner, repo, *_ = make(tmp_path, gateway=gw)
        j = job()
        await repo.create(j)
        runner._candidates.publish = _raise_cancelled  # the pod is stopped during the upload
        with pytest.raises(asyncio.CancelledError):
            await runner.run_job(j.id)
        assert (tmp_path / "jobs" / str(j.id) / "built.ok").exists()
        assert (await repo.get(j.id)).state == RUNNING


async def _raise_cancelled(*a, **k):
    raise asyncio.CancelledError


class TestUnsafeOutput:
    async def test_a_reserved_file_name_fails_the_job_at_once_instead_of_being_retried(
        self, tmp_path
    ):
        gw = FakeGateway(files={"final_taxonomy.json": b"{}", REPORT: b"{}"})
        runner, repo, _, s3, _ = make(tmp_path, gateway=gw)
        done = await run(runner, repo)
        assert done.state == FAILED and "reserved" in done.error
        assert not [p for p in s3.objects if not p.endswith("_claim.json")]


class TestNoBuildConfig:
    async def test_discover_is_refused_before_anything_is_read_or_claimed(self, tmp_path):
        runner, repo, store, s3, gw = make(tmp_path)
        gw.can_discover = False
        done = await run(runner, repo)
        assert done.state == FAILED and "no classifier build config" in done.error
        assert store.scrolls == 0 and gw.calls == [] and s3.objects == {}

    async def test_propose_still_runs(self, tmp_path):
        runner, repo, _, _, gw = make(tmp_path)
        gw.can_discover = False
        done = await run(runner, repo, kind=KIND_PROPOSE)
        assert done.state == COMPLETED


class TestCancel:
    async def test_cancel_before_the_build_writes_and_spends_nothing(self, tmp_path):
        runner, repo, _, s3, gw = make(tmp_path)
        j = job()
        await repo.create(j)
        repo.jobs[j.id].cancel_requested = True
        await runner.run_job(j.id)
        state = await repo.get(j.id)
        assert state.state == CANCELLED and gw.calls == [] and s3.objects == {}

    async def test_cancel_during_the_build_stops_before_the_upload(self, tmp_path):
        runner, repo, _, s3, gw = make(tmp_path, lease_seconds=3)
        gw.delay = 1.6
        j = job()
        await repo.create(j)
        task = asyncio.create_task(runner.run_job(j.id))
        await asyncio.sleep(0.2)
        await repo.request_cancel(j.id)
        await task
        state = await repo.get(j.id)
        assert state.state == CANCELLED
        assert not [p for p in s3.objects if not p.endswith("_claim.json")], (
            "no result was uploaded"
        )

    async def test_cancelling_an_unclaimed_job_is_immediate(self, tmp_path):
        repo = InMemoryTaxonomyJobRepository(Clock())
        j = job()
        await repo.create(j)
        assert (await repo.request_cancel(j.id)).state == CANCELLED


class TestCap:
    async def test_the_package_ledger_stopping_the_build_fails_the_job_without_a_result(
        self, tmp_path
    ):
        gw = FakeGateway()
        gw.error = SpendCapReached("embedding call refused: run cost cap would be exceeded")
        runner, repo, _, s3, _ = make(tmp_path, gateway=gw)
        done = await run(runner, repo)
        assert done.state == FAILED and "cost cap" in done.error
        assert not [p for p in s3.objects if not p.endswith("_claim.json")]

    async def test_the_job_passes_its_own_cap_to_the_package(self, tmp_path):
        runner, repo, _, _, gw = make(tmp_path)
        await run(runner, repo, spend_cap_usd=0.3)
        assert gw.calls[0]["spend_cap_usd"] == 0.3

    async def test_a_failing_package_fails_the_job_and_keeps_the_error(self, tmp_path):
        gw = FakeGateway()
        gw.error = RuntimeError("clustering blew up")
        runner, repo, _, s3, _ = make(tmp_path, gateway=gw)
        done = await run(runner, repo)
        assert done.state == FAILED and "clustering blew up" in done.error
        assert not (tmp_path / "jobs" / str(done.id)).exists()


class TestSmallCollections:
    async def test_too_little_text_fails_before_anything_is_paid_for(self, tmp_path):
        runner, repo, _, s3, gw = make(tmp_path, min_chunks=500)
        done = await run(runner, repo)
        assert done.state == FAILED
        assert "not enough text yet" in done.error and "at least 500" in done.error
        assert gw.calls == [] and s3.objects == {} and done.spend_used_usd == 0.0


class TestFailedRunsKeepTheirBill:
    async def test_a_failed_build_records_what_its_ledger_says(self, tmp_path):
        from airweave.domains.taxonomy.jobs.tests.helpers import FakeGateway

        class Dying(FakeGateway):
            def discover(self, chunks, *, work_dir, **kw):
                ledger = work_dir / "store" / "t" / "cost_ledger.json"
                ledger.parent.mkdir(parents=True)
                ledger.write_text(json.dumps({"total_usd": 0.0283}))
                raise IndexError("index -1 is out of bounds for axis 0 with size 0")

        runner, repo, *_ = make(tmp_path, gateway=Dying())
        done = await run(runner, repo)
        assert done.state == FAILED and "IndexError" in done.error
        assert done.spend_used_usd == pytest.approx(0.0283)

    async def test_a_call_in_flight_when_the_build_died_counts_at_its_worst_case(self, tmp_path):
        from airweave.domains.taxonomy.jobs.tests.helpers import FakeGateway

        class Dying(FakeGateway):
            def discover(self, chunks, *, work_dir, **kw):
                ledger = work_dir / "store" / "t" / "cost_ledger.json"
                ledger.parent.mkdir(parents=True)
                ledger.write_text(json.dumps({"total_usd": 0.02, "in_flight_usd": 0.005}))
                raise IndexError("died mid-call")

        runner, repo, *_ = make(tmp_path, gateway=Dying())
        done = await run(runner, repo)
        assert done.state == FAILED and done.spend_used_usd == pytest.approx(0.025)
