"""Starting, replaying, capping and scoping discover and propose jobs."""

from uuid import UUID

import pytest

from airweave.domains.taxonomy.jobs.records import (
    KIND_DISCOVER,
    KIND_PROPOSE,
    InMemoryTaxonomyJobRepository,
)
from airweave.domains.taxonomy.jobs.service import JobRequest, TaxonomyJobService, job_view
from airweave.domains.taxonomy.jobs.tests.helpers import COLLECTION, ORG, READABLE, TENANT
from airweave.domains.taxonomy.relabel.service import RelabelRefused


def req(**kw):
    base = {
        "kind": KIND_DISCOVER,
        "organization_id": ORG,
        "collection_id": COLLECTION,
        "collection_readable_id": READABLE,
        "tenant_id": TENANT,
        "candidate_version": "acme-first",
        "idempotency_key": "k1",
    }
    base.update(kw)
    return JobRequest(**base)


def make(discover=5.0, evolution=5.0):
    enqueued = []

    async def enqueue(job):
        enqueued.append(job.id)

    repo = InMemoryTaxonomyJobRepository()
    return (
        TaxonomyJobService(repo, enqueue, discover_cap_usd=discover, evolution_cap_usd=evolution),
        repo,
        enqueued,
    )


class TestCap:
    @pytest.mark.parametrize("kind", [KIND_DISCOVER, KIND_PROPOSE])
    async def test_default_cap_of_zero_approves_nothing(self, kind):
        svc, repo, enq = make(discover=0.0, evolution=0.0)
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(kind=kind))
        assert (e.value.status, e.value.code) == (422, "spend_cap_not_approved")
        assert repo.jobs == {} and enq == []

    async def test_each_kind_has_its_own_ceiling(self):
        svc, _, _ = make(discover=0.0, evolution=3.0)
        job, _ = await svc.start(req(kind=KIND_PROPOSE))
        assert job.spend_cap_usd == 3.0
        with pytest.raises(RelabelRefused):
            await svc.start(
                req(kind=KIND_DISCOVER, candidate_version="other", idempotency_key="k2")
            )

    async def test_a_request_can_lower_the_cap_never_raise_it(self):
        svc, _, _ = make(discover=5.0)
        low, _ = await svc.start(req(spend_cap_usd=1.5))
        assert low.spend_cap_usd == 1.5
        svc2, _, _ = make(discover=5.0)
        high, _ = await svc2.start(req(spend_cap_usd=500.0))
        assert high.spend_cap_usd == 5.0

    async def test_a_request_cannot_raise_a_zero_cap(self):
        svc, _, _ = make(discover=0.0)
        with pytest.raises(RelabelRefused):
            await svc.start(req(spend_cap_usd=100.0))

    async def test_a_zero_request_cap_refuses(self):
        svc, _, _ = make(discover=5.0)
        with pytest.raises(RelabelRefused):
            await svc.start(req(spend_cap_usd=0.0))


class TestValidation:
    @pytest.mark.parametrize(
        "kw, code",
        [
            ({"kind": "other"}, "bad_kind"),
            ({"candidate_version": ""}, "bad_version"),
            ({"candidate_version": "../x"}, "bad_version"),
            ({"idempotency_key": None}, "bad_key"),
            ({"idempotency_key": "a b"}, "bad_key"),
            ({"spend_cap_usd": -1.0}, "bad_cap"),
            ({"sample_size": 0}, "bad_sample_size"),
            ({"sample_size": 10**6}, "bad_sample_size"),
            ({"kind": KIND_PROPOSE, "sample_size": 10}, "bad_request"),
        ],
    )
    async def test_refused(self, kw, code):
        svc, repo, enq = make()
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(**kw))
        assert e.value.status == 422 and e.value.code == code and repo.jobs == {} and enq == []


class TestReplay:
    async def test_accepts_one_job_and_a_replay_returns_it(self):
        svc, repo, enq = make()
        first, created = await svc.start(req(sample_size=50))
        again, created2 = await svc.start(req(sample_size=50))
        assert created and not created2 and again.id == first.id
        # The job still waits for a worker, so the replay asks Temporal again; the workflow id
        # is the job id, so that start is a no-op when it is running already.
        assert enq == [first.id, first.id] and len(repo.jobs) == 1
        assert first.params["sample_size"] == 50 and len(first.params["fingerprint"]) == 64

    async def test_a_replay_after_the_job_finished_returns_it(self):
        svc, repo, enq = make()
        first, _ = await svc.start(req())
        repo.jobs[first.id].state = "COMPLETED"
        again, created = await svc.start(req())
        assert again.id == first.id and not created and enq == [first.id]

    async def test_same_key_other_candidate_is_a_conflict(self):
        svc, _, _ = make()
        await svc.start(req())
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(candidate_version="acme-other"))
        assert e.value.status == 409 and e.value.code == "idempotency_key_reused"

    async def test_a_different_job_while_one_runs_is_a_conflict_naming_it(self):
        svc, _, _ = make()
        first, _ = await svc.start(req())
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(idempotency_key="k2", candidate_version="acme-other"))
        assert e.value.status == 409 and e.value.extra["job_id"] == str(first.id)

    async def test_collections_are_independent(self):
        svc, _, _ = make()
        await svc.start(req())
        other, created = await svc.start(
            req(collection_id=UUID(int=99), collection_readable_id="neo-globex")
        )
        assert created

    async def test_allowed_again_after_the_job_ended(self):
        svc, repo, _ = make()
        first, _ = await svc.start(req())
        repo.jobs[first.id].state = "FAILED"
        second, created = await svc.start(
            req(idempotency_key="k2", candidate_version="acme-second")
        )
        assert created and second.id != first.id


class TestScope:
    async def test_get_and_cancel_are_organization_scoped(self):
        svc, _, _ = make()
        job, _ = await svc.start(req())
        other = UUID(int=777)
        assert (await svc.get(job.id, ORG)).id == job.id
        assert await svc.get(job.id, other) is None and await svc.cancel(job.id, other) is None
        assert (await svc.cancel(job.id, ORG)).state == "CANCELLED"

    def test_view_reports_status_words_and_the_candidate_prefix(self):
        from airweave.domains.taxonomy.jobs.tests.helpers import job as make_job

        j = make_job(
            state="COMPLETED", result={"candidate_prefix": "taxonomy/tenants/t/candidates/c/"}
        )
        view = job_view(j)
        assert view["status"] == "completed" and view["candidate_prefix"].endswith("candidates/c/")
        assert job_view(make_job(state="PENDING"))["status"] == "running"


class TestFiniteCaps:
    @pytest.mark.parametrize("cap", [float("nan"), float("inf"), float("-inf"), -0.5])
    async def test_a_cap_that_is_not_a_finite_amount_is_refused_before_anything_is_created(
        self, cap
    ):
        svc, repo, enq = make()
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(spend_cap_usd=cap))
        assert (e.value.status, e.value.code) == (422, "bad_cap")
        assert repo.jobs == {} and enq == []

    @pytest.mark.parametrize("ceiling", [float("nan"), float("inf"), -1.0])
    def test_a_ceiling_that_is_not_a_finite_amount_is_refused_at_construction(self, ceiling):
        repo = InMemoryTaxonomyJobRepository()

        async def enqueue(job):
            return None

        with pytest.raises(ValueError, match="spend cap"):
            TaxonomyJobService(repo, enqueue, discover_cap_usd=ceiling, evolution_cap_usd=1.0)
        with pytest.raises(ValueError, match="spend cap"):
            TaxonomyJobService(repo, enqueue, discover_cap_usd=1.0, evolution_cap_usd=ceiling)

    async def test_a_request_cap_below_the_ceiling_still_lowers_it(self):
        svc, _, _ = make(discover=5.0)
        job, _ = await svc.start(req(spend_cap_usd=1.5))
        assert job.spend_cap_usd == 1.5


class TestReplayMatchesTheWholeRequest:
    @pytest.mark.parametrize(
        "change",
        [
            {"sample_size": 99},
            {"corpus_description": "invoices"},
            {"spend_cap_usd": 1.0},
            {"tenant_id": "t-other"},
        ],
    )
    async def test_a_reused_key_with_other_parameters_is_a_conflict_not_a_replay(self, change):
        svc, repo, enq = make()
        first, _ = await svc.start(req(sample_size=50, spend_cap_usd=2.0))

        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(**{"sample_size": 50, "spend_cap_usd": 2.0, **change}))

        assert (e.value.status, e.value.code) == (409, "idempotency_key_reused")
        assert e.value.extra["job_id"] == str(first.id) and len(repo.jobs) == 1

    async def test_the_identical_request_is_still_a_replay(self):
        svc, _, _ = make()
        first, _ = await svc.start(req(sample_size=50, corpus_description="loans"))
        again, created = await svc.start(req(sample_size=50, corpus_description="loans"))
        assert again.id == first.id and created is False

    async def test_a_different_request_while_one_runs_names_the_running_job(self):
        svc, _, _ = make()
        first, _ = await svc.start(req(sample_size=50))
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(sample_size=60, idempotency_key="k2"))
        assert e.value.status == 409 and e.value.code == "job_active"
        assert e.value.extra["job_id"] == str(first.id)


class TestDispatchIsRecoverable:
    async def test_a_temporal_outage_does_not_fail_the_request_or_lose_the_job(self):
        repo = InMemoryTaxonomyJobRepository()

        async def down(job):
            raise ConnectionError("temporal is down")

        svc = TaxonomyJobService(repo, down, discover_cap_usd=5.0, evolution_cap_usd=5.0)

        job, created = await svc.start(req())

        assert created and job.state == "PENDING" and repo.jobs[job.id].state == "PENDING"

    async def test_replaying_the_request_starts_the_workflow_that_never_started(self):
        repo = InMemoryTaxonomyJobRepository()
        attempts = []

        async def flaky(job):
            attempts.append(job.id)
            if len(attempts) == 1:
                raise ConnectionError("temporal is down")

        svc = TaxonomyJobService(repo, flaky, discover_cap_usd=5.0, evolution_cap_usd=5.0)
        first, _ = await svc.start(req())
        again, created = await svc.start(req())

        assert again.id == first.id and not created
        assert attempts == [first.id, first.id]  # the replay started it

    async def test_a_running_job_is_not_dispatched_again_by_a_replay(self):
        svc, repo, enq = make()
        first, _ = await svc.start(req())
        repo.jobs[first.id].state = "RUNNING"

        await svc.start(req())

        assert enq == [first.id]


class TestTenantShape:
    @pytest.mark.parametrize("tenant", ["", "..", "../t-other", "a/b", ".hidden"])
    async def test_a_tenant_that_is_not_a_plain_name_is_refused(self, tenant):
        svc, repo, enq = make()
        with pytest.raises(RelabelRefused) as e:
            await svc.start(req(tenant_id=tenant))
        assert e.value.code == "bad_tenant" and repo.jobs == {} and enq == []
