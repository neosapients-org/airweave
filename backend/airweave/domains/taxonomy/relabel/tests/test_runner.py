"""The re-label job (A9): conditional writes, the sync race, replay and resume, the three modes."""

import asyncio
import json

import pytest

from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError
from airweave.domains.taxonomy.relabel.jobs import (
    CANCELLED,
    COMPLETED,
    FAILED,
    MODE_FULL,
    MODE_INCREMENTAL,
    RUNNING,
    SCOPE_NOT_LABELLED,
    ActiveJobExists,
    InMemoryJobRepository,
)
from airweave.domains.taxonomy.relabel.runner import RelabelRunner
from airweave.domains.taxonomy.relabel.tests.helpers import (
    COLLECTION,
    NEW,
    OLD,
    OTHER_COLLECTION,
    READABLE,
    Clock,
    FakeBus,
    FakeChunkStore,
    FakePool,
    FakeResolver,
    chunk,
    new_job,
    taxonomy,
)

LABEL_KEYS = {
    "taxonomy_status",
    "topic_ids",
    "broad_family_ids",
    "taxonomy_spans",
    "taxonomy_version",
    "taxonomy_piece_count",
    "taxonomy_admin_pieces",
    "taxonomy_unresolved_pieces",
}


def make(store, *, repo=None, pool=None, resolver=None, bus=None, clock=None, owner="w1", **kw):
    repo = repo or InMemoryJobRepository(clock or Clock())
    pool, resolver, bus = pool or FakePool(), resolver or FakeResolver(), bus or FakeBus()
    kw.setdefault("page_size", 3)
    kw.setdefault("batch_size", 2)
    runner = RelabelRunner(repo, lambda: store, resolver, pool, bus, owner=owner, **kw)
    return runner, repo, pool, resolver, bus


async def run(runner, repo, **job_kw):
    job = new_job(**job_kw)
    await repo.create(job)
    await runner.run_job(job.id)
    return await repo.get(job.id)


def old_chunks(n=5, version=OLD, status="ASSIGNED", doc="doc-1"):
    return [chunk(i, doc=doc, status=status, version=version) for i in range(n)]


class TestFullMode:
    async def test_relabels_chunks_at_another_version_and_skips_those_at_the_target(self):
        store = FakeChunkStore(
            [
                *old_chunks(4),
                chunk(9, doc="d2", status="ASSIGNED", version=NEW),
                chunk(10, doc="d3"),
            ]
        )  # no version at all
        runner, repo, pool, *_ = make(store)
        job = await run(runner, repo)
        assert job.state == COMPLETED and job.error is None
        assert job.chunks_relabelled == 5 and job.chunks_skipped_changed == 0
        assert {r["chunk_id"].split("/")[1] for r in pool.rows} == {
            k[1] for k in store.keys() if "doc-1" in k[1] or "d3" in k[1]
        }
        assert all(store.get(*k)["taxonomy_version"] == NEW for k in store.keys())
        assert store.closed

    async def test_label_fields_only_and_exactly_what_the_sync_writes(self):
        store = FakeChunkStore(old_chunks(2))
        before = {k: dict(v) for k, v in ((k, store.get(*k)) for k in store.keys())}
        runner, repo, *_ = make(store)
        await run(runner, repo)
        for _, fields, _ in store.updates:
            assert set(fields) == LABEL_KEYS
        for k in store.keys():
            for untouched in (
                "textual_representation",
                "airweave_system_metadata_hash",
                "chunk_char_start",
                "chunk_own_start",
                "chunk_char_end",
            ):
                assert store.get(*k)[untouched] == before[k][untouched]
            written = store.get(*k)
            assert written["topic_ids"] == {"acme:credit.loan_quality": 1}
            assert written["broad_family_ids"] == ["acme:credit"]
            spans = json.loads(written["taxonomy_spans"])
            assert spans and spans[0]["c"] == "acme:credit.loan_quality"

    async def test_classifier_sees_own_text_only(self):
        store = FakeChunkStore([chunk(0, version=OLD, status="ASSIGNED", body="Acme own text.")])
        runner, repo, pool, *_ = make(store)
        await run(runner, repo)
        assert [r["text"] for r in pool.rows] == ["Acme own text."]

    async def test_code_files_and_unpositioned_chunks_are_skipped_and_counted(self):
        store = FakeChunkStore(
            [
                chunk(0, schema="code_file_entity", version=OLD, status="ASSIGNED"),
                chunk(1, positions=False),
                chunk(2, version=OLD, status="ASSIGNED"),
            ]
        )
        runner, repo, pool, *_ = make(store)
        job = await run(runner, repo)
        assert (job.chunks_relabelled, job.chunks_unlabellable) == (1, 2)
        assert len(pool.rows) == 1
        assert (
            "taxonomy_version"
            not in store.get("code_file_entity", "code_file_entity_doc-1__chunk_0")
            or store.get("code_file_entity", "code_file_entity_doc-1__chunk_0")["taxonomy_version"]
            == OLD
        )

    async def test_error_and_not_labelled_chunks_are_included(self):
        store = FakeChunkStore(
            [chunk(0, status="ERROR"), chunk(1, status="NOT_LABELLED"), chunk(2)]
        )
        runner, repo, *_ = make(store)
        job = await run(runner, repo)
        assert job.chunks_relabelled == 3
        assert {store.get(*k)["taxonomy_status"] for k in store.keys()} == {"ASSIGNED"}

    async def test_blank_own_text_is_administrative_without_classifying(self):
        store = FakeChunkStore([chunk(0, body="   ", version=OLD, status="ASSIGNED")])
        runner, repo, pool, *_ = make(store)
        job = await run(runner, repo)
        assert job.chunks_relabelled == 1 and pool.calls == 0
        written = store.get("file_entity", "file_entity_doc-1__chunk_0")
        assert written["taxonomy_status"] == "ADMINISTRATIVE" and written["taxonomy_version"] == NEW


class TestTestAndSet:
    async def test_every_update_carries_the_hash_that_was_read(self):
        store = FakeChunkStore([chunk(0, hash_="aaa"), chunk(1, hash_="bbb")])
        runner, repo, *_ = make(store)
        await run(runner, repo)
        assert sorted(h for _, _, h in store.updates) == ["aaa", "bbb"]

    async def test_a_sync_replacing_a_chunk_during_the_relabel_wins(self):
        store = FakeChunkStore([chunk(0, body="Old Acme text.", hash_="old-hash")])
        key = ("file_entity", "file_entity_doc-1__chunk_0")

        async def sync_replaces_chunk(schema, doc_id):
            # the sync rewrites the chunk with new text, a new hash and target labels
            store.docs[key].update(
                textual_representation="header\nNew Acme text.",
                airweave_system_metadata_hash="new-hash",
                taxonomy_status="UNRESOLVED",
                taxonomy_version=NEW,
                topic_ids={},
            )
            store.before_update = None

        store.before_update = sync_replaces_chunk
        runner, repo, *_ = make(store)
        job = await run(runner, repo)
        assert job.chunks_skipped_changed == 1 and job.chunks_relabelled == 0
        kept = store.get(*key)
        assert kept["taxonomy_status"] == "UNRESOLVED" and kept["topic_ids"] == {}
        assert kept["textual_representation"].endswith("New Acme text.")
        assert store.updates == []  # no old-text label landed on the new text

    async def test_a_chunk_deleted_meanwhile_is_skipped_not_failed(self):
        store = FakeChunkStore([chunk(0)])

        async def deleted(schema, doc_id):
            store.docs.pop((schema, doc_id))

        store.before_update = deleted
        runner, repo, *_ = make(store)
        job = await run(runner, repo)
        assert job.state == COMPLETED and job.chunks_skipped_changed == 1 and job.chunks_failed == 0

    async def test_a_chunk_without_a_hash_is_never_written(self):
        store = FakeChunkStore([chunk(0, hash_=None)])
        runner, repo, *_ = make(store)
        job = await run(runner, repo)
        assert store.updates == [] and job.chunks_failed == 1

    async def test_control_without_the_race_the_label_lands(self):
        store = FakeChunkStore([chunk(0)])
        runner, repo, *_ = make(store)
        assert (await run(runner, repo)).chunks_relabelled == 1


class TestReplayAndResume:
    async def test_running_a_finished_job_again_does_nothing(self):
        store = FakeChunkStore(old_chunks(3))
        runner, repo, pool, *_ = make(store)
        job = await run(runner, repo)
        calls, updates = pool.calls, len(store.updates)
        await runner.run_job(job.id)
        assert pool.calls == calls and len(store.updates) == updates
        assert (await repo.get(job.id)).state == COMPLETED

    async def test_a_second_job_for_the_same_target_finds_nothing_to_do(self):
        store = FakeChunkStore(old_chunks(3))
        runner, repo, pool, *_ = make(store)
        await run(runner, repo)
        calls = pool.calls
        second = await run(runner, repo, idempotency_key="again")
        assert second.state == COMPLETED and second.chunks_relabelled == 0 and pool.calls == calls

    async def test_a_restart_resumes_from_the_checkpoint(self):
        clock = Clock()
        store = FakeChunkStore(old_chunks(8))
        repo = InMemoryJobRepository(clock)
        pool1 = FakePool()
        seen = []

        def die_on_second_page():
            seen.append(1)
            if len(seen) == 2:
                raise asyncio.CancelledError  # the worker is stopped mid-job

        pool1.on_call = die_on_second_page
        r1, *_ = make(store, repo=repo, pool=pool1, owner="w1", page_size=2, batch_size=2)
        job = new_job()
        await repo.create(job)
        with pytest.raises(asyncio.CancelledError):
            await r1.run_job(job.id)
        mid = await repo.get(job.id)
        assert mid.state == RUNNING and mid.checkpoint and mid.chunks_relabelled == 2

        # another worker cannot take it while the lease is live, and can once it expires
        r2, _, pool2, *_ = make(
            store, repo=repo, owner="w2", clock=clock, page_size=2, batch_size=2
        )
        await r2.run_job(job.id)
        assert pool2.calls == 0
        clock.advance(121)
        await r2.run_job(job.id)
        done = await repo.get(job.id)
        assert done.state == COMPLETED
        assert len(pool2.rows) == 6, "the two chunks done before the restart are not redone"
        assert all(store.get(*k)["taxonomy_version"] == NEW for k in store.keys())

    async def test_an_unusable_checkpoint_restarts_the_scan(self):
        store = FakeChunkStore(old_chunks(3))
        runner, repo, *_ = make(store)
        job = await run(runner, repo, checkpoint="garbage")
        assert job.state == COMPLETED and job.chunks_relabelled == 3

    async def test_a_worker_that_lost_its_lease_stops_without_finishing(self):
        clock = Clock()
        store = FakeChunkStore(old_chunks(6))
        repo = InMemoryJobRepository(clock)
        pool = FakePool()
        runner, *_ = make(store, repo=repo, pool=pool, owner="w1", clock=clock, page_size=2)
        job = new_job()
        await repo.create(job)

        def lease_taken_by_another():
            clock.advance(500)
            asyncio.get_event_loop().create_task(repo.claim(job.id, "w2", 120))

        pool.on_call = lease_taken_by_another
        await runner.run_job(job.id)
        state = await repo.get(job.id)
        assert state.lease_owner == "w2" and state.state == RUNNING


class TestIncremental:
    def chunks(self):
        return [
            chunk(0, status="ASSIGNED", version=OLD),
            chunk(1, status="ADMINISTRATIVE", version=OLD),
            chunk(2, status="UNRESOLVED", version=OLD),
            chunk(3, status="PARTIALLY_ASSIGNED", version=OLD),
            chunk(4, status="ERROR"),
            chunk(5, status="NOT_LABELLED"),
            chunk(6, status="ASSIGNED", version=NEW),
        ]

    async def test_classifies_the_backlog_and_stamps_the_rest(self):
        store = FakeChunkStore(self.chunks())
        runner, repo, pool, *_ = make(store)
        job = await run(runner, repo, mode=MODE_INCREMENTAL)
        assert job.state == COMPLETED
        assert (job.chunks_relabelled, job.chunks_stamped) == (4, 2)
        classified = {r["chunk_id"].split("__chunk_")[1] for r in pool.rows}
        assert classified == {"2", "3", "4", "5"}, "carried-over chunks are not classified"

    async def test_stamping_writes_the_version_only_with_the_hash_condition(self):
        store = FakeChunkStore([chunk(0, status="ASSIGNED", version=OLD, hash_="hh")])
        runner, repo, pool, *_ = make(store)
        await run(runner, repo, mode=MODE_INCREMENTAL)
        assert store.updates == [("file_entity_doc-1__chunk_0", {"taxonomy_version": NEW}, "hh")]
        assert pool.calls == 0

    async def test_carried_over_labels_are_kept(self):
        store = FakeChunkStore([chunk(0, status="ASSIGNED", version=OLD, topic_ids={"acme:x": 1})])
        runner, repo, *_ = make(store)
        await run(runner, repo, mode=MODE_INCREMENTAL)
        kept = store.get("file_entity", "file_entity_doc-1__chunk_0")
        assert kept["topic_ids"] == {"acme:x": 1} and kept["taxonomy_status"] == "ASSIGNED"

    async def test_nothing_is_left_at_another_version(self):
        store = FakeChunkStore(self.chunks())
        runner, repo, *_ = make(store)
        await run(runner, repo, mode=MODE_INCREMENTAL)
        from airweave.domains.taxonomy.relabel.stats import collect_stats

        stats = await collect_stats(store, COLLECTION, NEW)
        assert stats.other_version == 0

    async def test_stamp_loses_to_a_sync_too(self):
        store = FakeChunkStore([chunk(0, status="ASSIGNED", version=OLD, hash_="old")])

        async def sync(schema, doc_id):
            store.docs[(schema, doc_id)].update(
                airweave_system_metadata_hash="new", taxonomy_version=NEW
            )

        store.before_update = sync
        runner, repo, *_ = make(store)
        job = await run(runner, repo, mode=MODE_INCREMENTAL)
        assert job.chunks_skipped_changed == 1 and job.chunks_stamped == 0

    async def test_full_mode_reclassifies_everything_instead(self):
        store = FakeChunkStore(self.chunks())
        runner, repo, pool, *_ = make(store)
        job = await run(runner, repo, mode=MODE_FULL)
        assert job.chunks_relabelled == 6 and job.chunks_stamped == 0


class TestNotLabelledScope:
    async def test_labels_only_not_labelled_chunks(self):
        store = FakeChunkStore(
            [
                chunk(0),
                chunk(1, status="NOT_LABELLED"),
                chunk(2, status="ASSIGNED", version=OLD),
                chunk(3, status="ERROR"),
            ]
        )
        runner, repo, pool, *_ = make(store)
        job = await run(runner, repo, scope=SCOPE_NOT_LABELLED, previous_version=None)
        assert job.state == COMPLETED and job.chunks_relabelled == 2
        assert store.get("file_entity", "file_entity_doc-1__chunk_2")["taxonomy_version"] == OLD
        assert store.get("file_entity", "file_entity_doc-1__chunk_3")["taxonomy_status"] == "ERROR"

    async def test_chunks_labelled_by_a_sync_meanwhile_are_not_touched(self):
        store = FakeChunkStore([chunk(0, hash_="a")])

        async def sync(schema, doc_id):
            store.docs[(schema, doc_id)].update(
                airweave_system_metadata_hash="b", taxonomy_status="ASSIGNED", taxonomy_version=NEW
            )

        store.before_update = sync
        runner, repo, *_ = make(store)
        job = await run(runner, repo, scope=SCOPE_NOT_LABELLED, previous_version=None)
        assert job.chunks_skipped_changed == 1 and store.updates == []

    async def test_works_when_the_collection_has_no_prior_version(self):
        store = FakeChunkStore([chunk(i) for i in range(4)])
        runner, repo, *_ = make(store)
        job = await run(runner, repo, scope=SCOPE_NOT_LABELLED, previous_version=None)
        assert job.chunks_relabelled == 4


class TestStopsVisibly:
    async def test_target_that_is_not_the_active_labelling_version(self):
        store = FakeChunkStore(old_chunks(2))
        runner, repo, pool, *_ = make(store, resolver=FakeResolver(taxonomy(OLD)))
        job = await run(runner, repo)
        assert job.state == FAILED and "not the job's target" in job.error
        assert pool.calls == 0 and store.updates == []

    async def test_no_active_file(self):
        runner, repo, *_ = make(FakeChunkStore(old_chunks(1)), resolver=FakeResolver("none"))
        job = await run(runner, repo)
        assert job.state == FAILED and "no active taxonomy" in job.error

    async def test_unusable_taxonomy(self):
        runner, repo, *_ = make(
            FakeChunkStore(old_chunks(1)),
            resolver=FakeResolver(TaxonomyResolutionError("checksum mismatch")),
        )
        job = await run(runner, repo)
        assert job.state == FAILED and "checksum mismatch" in job.error

    async def test_resolved_with_a_fresh_read(self):
        runner, repo, _, resolver, _ = make(FakeChunkStore(old_chunks(1)))
        await run(runner, repo)
        assert resolver.calls[0] == (READABLE, True)

    async def test_active_file_changing_mid_job_stops_it(self):
        store = FakeChunkStore(old_chunks(6))
        resolver = FakeResolver()
        pool = FakePool()
        pool.on_call = lambda: setattr(resolver, "result", taxonomy(OLD))  # rolled back meanwhile
        runner, repo, *_ = make(store, pool=pool, resolver=resolver, page_size=2)
        job = await run(runner, repo)
        assert job.state == FAILED and "not the job's target" in job.error
        assert job.chunks_relabelled == 2

    async def test_cancel_stops_the_job_and_sends_no_event(self):
        store = FakeChunkStore(old_chunks(6))
        pool = FakePool()
        runner, repo, _, _, bus = make(store, pool=pool, page_size=2)
        job = new_job()
        await repo.create(job)
        pool.on_call = lambda: asyncio.get_event_loop().create_task(repo.request_cancel(job.id))
        await runner.run_job(job.id)
        done = await repo.get(job.id)
        assert done.state == CANCELLED and done.chunks_relabelled < 6
        assert bus.events == []

    async def test_cancelling_a_job_nobody_runs(self):
        repo = InMemoryJobRepository(Clock())
        job = new_job()
        await repo.create(job)
        assert (await repo.request_cancel(job.id)).state == CANCELLED

    async def test_spend_cap_reached_fails_the_job(self):
        store = FakeChunkStore(old_chunks(20))
        runner, repo, *_ = make(store, page_size=5, usd_per_thousand=1000.0)  # 1 USD per chunk
        job = await run(runner, repo, spend_cap_usd=7.0)
        assert job.state == FAILED and "spend cap" in job.error
        assert job.chunks_relabelled == 10

    async def test_classifier_failing_leaves_chunks_unchanged_and_gives_up(self):
        store = FakeChunkStore(old_chunks(12))
        runner, repo, *_ = make(store, pool=FakePool(fail=RuntimeError("down")), page_size=2)
        job = await run(runner, repo)
        assert job.state == FAILED and "in a row" in job.error
        assert job.chunks_failed >= 6
        assert all(store.get(*k)["taxonomy_version"] == OLD for k in store.keys())

    async def test_one_bad_batch_does_not_stop_the_job(self):
        store = FakeChunkStore(old_chunks(6))
        pool = FakePool()
        first = pool.classify
        state = {"n": 0}

        async def flaky(tax, rows, timeout):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("blip")
            return await first(tax, rows, timeout)

        pool.classify = flaky
        runner, repo, *_ = make(store, pool=pool, page_size=2, batch_size=2)
        job = await run(runner, repo)
        assert job.state == COMPLETED and job.chunks_failed == 2 and job.chunks_relabelled == 4

    async def test_classifier_dropping_a_chunk_counts_it_failed(self):
        store = FakeChunkStore(old_chunks(2))
        runner, repo, *_ = make(
            store, pool=FakePool(drop=["file_entity/file_entity_doc-1__chunk_0"])
        )
        job = await run(runner, repo)
        assert (job.chunks_relabelled, job.chunks_failed) == (1, 1)

    async def test_vespa_down_on_every_write_fails_the_job(self):
        store = FakeChunkStore(old_chunks(12))
        store.fail_updates = RuntimeError("vespa unavailable")
        runner, repo, *_ = make(store, page_size=2)
        job = await run(runner, repo)
        assert job.state == FAILED and "vespa unavailable" in job.error


class TestEvent:
    async def test_completed_event_has_the_contract_fields(self):
        store = FakeChunkStore(old_chunks(3))
        runner, repo, _, _, bus = make(store)
        job = await run(runner, repo, mode=MODE_INCREMENTAL)
        (event,) = bus.events
        body = event.model_dump(mode="json")
        assert body["event_type"] == "taxonomy.relabel.completed"
        assert body["relabel_job_id"] == str(job.id) and body["collection_id"] == str(COLLECTION)
        assert body["collection_readable_id"] == READABLE
        assert (body["taxonomy_version"], body["previous_taxonomy_version"]) == (NEW, OLD)
        assert (body["chunks_relabelled"], body["chunks_failed"]) == (3, 0)
        assert body["status"] == "completed" and body["error"] is None
        assert body["organization_id"]

    async def test_failed_job_sends_a_failed_event_with_the_error(self):
        runner, repo, _, _, bus = make(FakeChunkStore(old_chunks(1)), resolver=FakeResolver("none"))
        await run(runner, repo)
        assert bus.events[0].status == "failed" and "no active taxonomy" in bus.events[0].error

    async def test_event_not_sent_twice(self):
        runner, repo, _, _, bus = make(FakeChunkStore(old_chunks(1)))
        await run(runner, repo)
        await runner.sweep_once()
        await runner.sweep_once()
        assert len(bus.events) == 1

    async def test_a_lost_event_is_sent_by_the_next_sweep(self):
        runner, repo, _, _, bus = make(FakeChunkStore(old_chunks(1)))
        bus.fail = True
        job = await run(runner, repo)
        assert job.state == COMPLETED and bus.events == []
        bus.fail = False
        await runner.sweep_once()
        assert len(bus.events) == 1 and (await repo.get(job.id)).completion_event_sent


class TestSweep:
    async def test_a_sweep_runs_pending_jobs(self):
        store = FakeChunkStore(old_chunks(3))
        runner, repo, *_ = make(store)
        job = new_job()
        await repo.create(job)
        assert await runner.sweep_once() == [job.id]
        await runner.wait_idle()
        assert (await repo.get(job.id)).state == COMPLETED

    async def test_a_job_already_running_here_is_not_started_twice(self):
        store = FakeChunkStore(old_chunks(3))
        runner, repo, pool, *_ = make(store)
        job = new_job()
        await repo.create(job)
        await runner.sweep_once()
        assert await runner.sweep_once() == []
        await runner.wait_idle()

    async def test_run_forever_stops_on_the_event(self):
        runner, repo, *_ = make(FakeChunkStore(old_chunks(2)), resume_interval=0.05)
        stop = asyncio.Event()
        task = asyncio.create_task(runner.run_forever(stop))
        job = new_job()
        await repo.create(job)
        for _ in range(100):
            await asyncio.sleep(0.02)
            if (await repo.get(job.id)).state == COMPLETED:
                break
        stop.set()
        await asyncio.wait_for(task, 2)
        assert (await repo.get(job.id)).state == COMPLETED


class TestOnePerCollection:
    async def test_second_job_for_the_collection_is_refused_while_one_is_active(self):
        repo = InMemoryJobRepository(Clock())
        first = new_job()
        await repo.create(first)
        with pytest.raises(ActiveJobExists) as e:
            await repo.create(new_job())
        assert e.value.job.id == first.id

    async def test_other_collections_are_independent(self):
        repo = InMemoryJobRepository(Clock())
        await repo.create(new_job())
        await repo.create(
            new_job(collection_id=OTHER_COLLECTION, collection_readable_id="neo-globex")
        )

    async def test_allowed_again_after_the_job_ended(self):
        store = FakeChunkStore(old_chunks(1))
        runner, repo, *_ = make(store)
        await run(runner, repo)
        await repo.create(new_job())
