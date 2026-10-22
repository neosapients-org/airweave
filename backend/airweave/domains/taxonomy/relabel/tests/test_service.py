"""Accepting jobs: validation, the spend cap before the start, replays, one job per collection."""

import pytest

from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError
from airweave.domains.taxonomy.relabel.jobs import (
    MODE_INCREMENTAL,
    SCOPE_NOT_LABELLED,
    InMemoryJobRepository,
)
from airweave.domains.taxonomy.relabel.service import RelabelRefused, RelabelService, StartRequest
from airweave.domains.taxonomy.relabel.tests.helpers import (
    COLLECTION,
    NEW,
    OLD,
    ORG,
    READABLE,
    Clock,
    FakeChunkStore,
    FakeResolver,
    chunk,
    taxonomy,
)


def req(**kw):
    base = {
        "organization_id": ORG,
        "collection_id": COLLECTION,
        "collection_readable_id": READABLE,
        "taxonomy_version": NEW,
    }
    base.update(kw)
    return StartRequest(**base)


def service(store=None, cap=10.0, rate=0.006, repo=None, resolver=None):
    store = store or FakeChunkStore([chunk(i, status="ASSIGNED", version=OLD) for i in range(5)])
    repo = repo or InMemoryJobRepository(Clock())
    svc = RelabelService(
        repo,
        lambda: store,
        configured_cap_usd=cap,
        usd_per_thousand=rate,
        resolver=resolver or FakeResolver(),
    )
    return svc, repo


class TestTheTargetIsTheActiveVersion:
    async def test_a_version_that_is_not_active_is_refused_before_the_scan(self):
        store = FakeChunkStore([chunk(i, status="ASSIGNED", version=OLD) for i in range(3)])
        svc, repo = service(store, resolver=FakeResolver(taxonomy(OLD)))
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req())
        assert (e.value.status, e.value.code) == (409, "not_the_active_version")
        assert e.value.extra["active_version"] == OLD
        assert store.scrolls == 0 and repo.jobs == {}

    async def test_a_collection_with_no_active_taxonomy_is_refused(self):
        store = FakeChunkStore([chunk(0, status="ASSIGNED", version=OLD)])
        svc, repo = service(store, resolver=FakeResolver("none"))
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req())
        assert e.value.code == "no_active_taxonomy" and store.scrolls == 0

    async def test_an_unreadable_active_file_is_a_503_not_a_500(self):
        svc, _ = service(resolver=FakeResolver(TaxonomyResolutionError("s3 down")))
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req())
        assert (e.value.status, e.value.code) == (503, "taxonomy_unreadable")

    async def test_the_active_file_is_read_fresh(self):
        resolver = FakeResolver()
        svc, _ = service(resolver=resolver)
        await svc.start(req())
        assert resolver.calls == [(READABLE, True)]


class TestSpendCap:
    async def test_default_cap_of_zero_refuses_every_job(self):
        svc, repo = service(cap=0.0)
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req())
        assert (e.value.status, e.value.code) == (422, "spend_cap_not_approved")
        assert repo.jobs == {}

    async def test_zero_cap_allows_a_stamp_only_incremental_job(self):
        # carried-over chunks only: nothing is classified, so nothing calls the embedder
        store = FakeChunkStore([chunk(i, status="ASSIGNED", version=OLD) for i in range(4)])
        svc, repo = service(store, cap=0.0)
        job, created = await svc.start(req(mode=MODE_INCREMENTAL))
        assert created and job.spend_estimate_usd == 0 and job.chunks_total == 4
        assert job.spend_cap_usd == 0.0 and len(repo.jobs) == 1

    async def test_zero_cap_still_refuses_an_incremental_job_with_a_backlog(self):
        store = FakeChunkStore(
            [chunk(0, status="UNRESOLVED", version=OLD), chunk(1, status="ASSIGNED", version=OLD)]
        )
        svc, repo = service(store, cap=0.0)
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(mode=MODE_INCREMENTAL))
        assert e.value.code == "spend_cap_not_approved" and repo.jobs == {}

    async def test_zero_cap_still_refuses_full_and_not_labelled_jobs(self):
        for kw in ({}, {"scope": SCOPE_NOT_LABELLED}):
            store = FakeChunkStore([chunk(0), chunk(1, status="ASSIGNED", version=OLD)])
            svc, _ = service(store, cap=0.0)
            with pytest.raises(RelabelRefused):
                await svc.start(req(**kw))

    async def test_a_request_cannot_raise_a_zero_cap(self):
        store = FakeChunkStore([chunk(0, status="UNRESOLVED", version=OLD)])
        svc, _ = service(store, cap=0.0)
        with pytest.raises(RelabelRefused):
            await svc.start(req(mode=MODE_INCREMENTAL, spend_cap_usd=100.0))

    async def test_a_stamp_only_job_with_nothing_to_do_is_allowed_at_zero(self):
        svc, _ = service(FakeChunkStore([]), cap=0.0)
        job, created = await svc.start(req(mode=MODE_INCREMENTAL))
        assert created and job.chunks_total == 0

    async def test_estimate_above_the_cap_is_refused_before_anything_is_written(self):
        store = FakeChunkStore([chunk(i, status="ASSIGNED", version=OLD) for i in range(5)])
        svc, repo = service(store, cap=0.01, rate=10.0)  # 5 chunks = USD 0.05
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req())
        assert e.value.code == "spend_cap_exceeded" and e.value.extra["estimate_usd"] == 0.05
        assert repo.jobs == {} and store.updates == []

    async def test_estimate_within_the_cap_is_accepted_and_recorded(self):
        svc, _ = service(rate=0.006)
        job, created = await svc.start(req())
        assert created and job.spend_estimate_usd == 0.00003 and job.spend_cap_usd == 10.0
        assert job.state == "PENDING" and job.chunks_total == 5 and job.previous_version == OLD

    async def test_a_request_can_lower_the_cap_but_not_raise_it(self):
        svc, _ = service(cap=10.0, rate=1000.0)  # 5 chunks cost USD 5
        with pytest.raises(RelabelRefused):
            await svc.start(req(spend_cap_usd=1.0))
        job, _ = await svc.start(req(spend_cap_usd=1000.0))
        assert job.spend_cap_usd == 10.0

    async def test_incremental_estimate_counts_only_the_backlog(self):
        chunks = [chunk(0, status="UNRESOLVED", version=OLD)] + [
            chunk(i, status="ASSIGNED", version=OLD) for i in range(1, 6)
        ]
        svc, _ = service(FakeChunkStore(chunks), cap=1.0, rate=1000.0)  # USD 1 per chunk
        job, _ = await svc.start(req(mode=MODE_INCREMENTAL))
        assert job.spend_estimate_usd == 1.0 and job.chunks_total == 6  # 1 classified, 5 stamped

    async def test_not_labelled_scope_estimates_the_not_labelled_chunks(self):
        chunks = [chunk(0), chunk(1), chunk(2, status="ASSIGNED", version=OLD)]
        svc, _ = service(FakeChunkStore(chunks), rate=1000.0)
        job, _ = await svc.start(req(scope=SCOPE_NOT_LABELLED))
        assert job.spend_estimate_usd == 2.0 and job.previous_version == OLD


class TestValidation:
    @pytest.mark.parametrize(
        "kw, code",
        [
            ({"taxonomy_version": ""}, "bad_version"),
            ({"taxonomy_version": "../x"}, "bad_version"),
            ({"mode": "fast"}, "bad_mode"),
            ({"scope": "everything"}, "bad_scope"),
            ({"scope": SCOPE_NOT_LABELLED, "mode": MODE_INCREMENTAL}, "bad_mode"),
            ({"idempotency_key": "a b"}, "bad_key"),
            ({"spend_cap_usd": -1.0}, "bad_cap"),
        ],
    )
    async def test_refused(self, kw, code):
        svc, repo = service()
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(**kw))
        assert e.value.status == 422 and e.value.code == code and repo.jobs == {}


class TestReplayAndOneJob:
    async def test_a_replayed_start_returns_the_job(self):
        svc, repo = service()
        first, created1 = await svc.start(req(idempotency_key="mig-1-attempt-1"))
        again, created2 = await svc.start(req(idempotency_key="mig-1-attempt-1"))
        assert created1 and not created2 and again.id == first.id and len(repo.jobs) == 1

    async def test_a_replay_after_the_job_finished_still_returns_it(self):
        svc, repo = service()
        first, _ = await svc.start(req(idempotency_key="k"))
        repo.jobs[first.id].state = "COMPLETED"
        again, created = await svc.start(req(idempotency_key="k"))
        assert again.id == first.id and not created

    async def test_same_key_other_parameters_is_a_conflict(self):
        svc, _ = service()
        await svc.start(req(idempotency_key="k"))
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(idempotency_key="k", mode=MODE_INCREMENTAL))
        assert e.value.status == 409 and e.value.code == "idempotency_key_reused"

    async def test_identical_start_while_running_returns_the_running_job(self):
        svc, _ = service()
        first, _ = await svc.start(req())
        again, created = await svc.start(req(idempotency_key="other-key"))
        assert again.id == first.id and not created

    async def test_a_different_start_while_one_runs_is_a_conflict_naming_the_job(self):
        svc, _ = service()
        first, _ = await svc.start(req())
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(taxonomy_version="acme-2027.01"))
        assert e.value.status == 409 and e.value.extra["relabel_job_id"] == str(first.id)

    async def test_the_next_job_is_allowed_after_the_first_ended(self):
        svc, repo = service()
        first, _ = await svc.start(req())
        repo.jobs[first.id].state = "COMPLETED"
        second, created = await svc.start(req(idempotency_key="again"))
        assert created and second.id != first.id

    async def test_collections_are_independent(self):
        from uuid import UUID

        svc, _ = service()
        await svc.start(req())
        other, created = await svc.start(
            req(collection_id=UUID(int=99), collection_readable_id="neo-globex")
        )
        assert created

    async def test_store_is_closed_after_the_scan(self):
        store = FakeChunkStore([chunk(0)])
        svc, _ = service(store)
        await svc.start(req())
        assert store.closed


class TestScope:
    async def test_get_and_cancel_are_organization_scoped(self):
        from uuid import UUID

        svc, _ = service()
        job, _ = await svc.start(req())
        assert (await svc.get(job.id, ORG)).id == job.id
        assert await svc.get(job.id, UUID(int=777)) is None
        assert await svc.cancel(job.id, UUID(int=777)) is None
        assert (await svc.cancel(job.id, ORG)).state == "CANCELLED"


class TestFiniteCaps:
    @pytest.mark.parametrize("cap", [float("nan"), float("inf"), float("-inf")])
    async def test_a_cap_that_is_not_a_finite_amount_is_refused(self, cap):
        svc, repo = service()
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(spend_cap_usd=cap))
        assert e.value.status == 422 and e.value.code == "bad_cap" and repo.jobs == {}

    @pytest.mark.parametrize("ceiling", [float("nan"), float("inf"), -1.0])
    def test_a_configured_ceiling_that_is_not_a_finite_amount_is_refused(self, ceiling):
        with pytest.raises(ValueError, match="spend cap"):
            RelabelService(
                InMemoryJobRepository(Clock()),
                lambda: None,
                configured_cap_usd=ceiling,
                usd_per_thousand=0.006,
                resolver=FakeResolver(),
            )

    @pytest.mark.parametrize(
        "requested, configured, expected",
        [
            (None, 5.0, 5.0),
            (2.0, 5.0, 2.0),
            (9.0, 5.0, 5.0),
            (None, float("nan"), 0.0),
            (1.0, float("nan"), 0.0),
            (float("nan"), 5.0, 0.0),
            (-1.0, 5.0, 0.0),
        ],
    )
    def test_effective_cap_never_returns_a_value_a_comparison_cannot_trip_on(
        self, requested, configured, expected
    ):
        from airweave.domains.taxonomy.relabel.planning import effective_cap

        assert effective_cap(requested, configured) == expected


class TestReplayIncludesTheCap:
    async def test_the_same_key_with_a_different_cap_is_a_conflict(self):
        svc, _ = service(cap=10.0)
        await svc.start(req(idempotency_key="k", spend_cap_usd=2.0))
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(idempotency_key="k", spend_cap_usd=5.0))
        assert e.value.status == 409 and e.value.code == "idempotency_key_reused"

    async def test_the_same_key_and_cap_is_a_replay(self):
        svc, repo = service(cap=10.0)
        first, _ = await svc.start(req(idempotency_key="k", spend_cap_usd=2.0))
        again, created = await svc.start(req(idempotency_key="k", spend_cap_usd=2.0))
        assert again.id == first.id and not created and len(repo.jobs) == 1

    async def test_a_cap_above_the_ceiling_is_the_ceiling_so_it_matches_a_start_without_one(self):
        svc, _ = service(cap=3.0)
        first, _ = await svc.start(req(idempotency_key="k"))
        again, created = await svc.start(req(idempotency_key="k", spend_cap_usd=99.0))
        assert again.id == first.id and not created
