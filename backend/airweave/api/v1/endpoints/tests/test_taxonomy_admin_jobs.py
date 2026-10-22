"""The discover and propose endpoints: caps, the tenant a discovery writes to, replays."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from airweave.api.v1.endpoints import taxonomy_admin as ep
from airweave.api.v1.endpoints.tests.test_taxonomy_admin_relabel import CollectionRepo, ctx
from airweave.domains.taxonomy.jobs.records import InMemoryTaxonomyJobRepository
from airweave.domains.taxonomy.jobs.service import TaxonomyJobService, resolve_tenant
from airweave.domains.taxonomy.relabel.service import RelabelRefused


def job_service(cap=5.0):
    enqueued = []

    async def enqueue(job):
        enqueued.append(job.id)

    repo = InMemoryTaxonomyJobRepository()
    return TaxonomyJobService(repo, enqueue, cap_usd=cap), enqueued


class Resolver:
    def __init__(self, tenant=None):
        self.tenant = tenant

    async def resolve(self, readable_id, fresh=False):
        return SimpleNamespace(tenant_id=self.tenant) if self.tenant else None


@pytest.fixture
def wired(monkeypatch):
    def wire(active_tenant=None):
        monkeypatch.setattr(ep, "_serving_tenant", lambda container: Resolver(active_tenant))

    return wire


async def discover(svc, c, **body):
    payload = {"candidate_version": "acme-first", "idempotency_key": "k1", **body}
    return await ep.start_taxonomy_discovery(
        readable_id="neo-acme",
        body=ep.DiscoveryRequest(**payload),
        db=AsyncMock(),
        ctx=c,
        collection_repo=CollectionRepo(),
        service=svc,
    )


class TestSpendCapOnTheWire:
    @pytest.mark.parametrize("cap", [float("nan"), float("inf"), -1.0])
    def test_a_cap_that_is_not_a_finite_amount_is_refused_by_every_request_model(self, cap):
        for model, extra in (
            (ep.DiscoveryRequest, {}),
            (ep.ProposalRequest, {}),
            (ep.RelabelRequest, {"taxonomy_version": "acme-2026.11"}),
        ):
            with pytest.raises(ValidationError):
                model(candidate_version="v", idempotency_key="k", spend_cap_usd=cap, **extra)

    def test_json_nan_is_refused_too(self):
        with pytest.raises(ValidationError):
            ep.DiscoveryRequest.model_validate_json(
                '{"candidate_version":"v","idempotency_key":"k","spend_cap_usd":NaN}'
            )


class TestTheTenantOfADiscovery:
    def test_a_first_discovery_takes_the_named_tenant(self):
        assert resolve_tenant(None, "t-acme") == "t-acme"

    def test_without_a_name_the_active_taxonomy_names_the_tenant(self):
        assert resolve_tenant("t-acme", None) == "t-acme"

    def test_a_name_that_matches_the_active_taxonomy_is_accepted(self):
        assert resolve_tenant("t-acme", "t-acme") == "t-acme"

    def test_a_name_that_differs_from_the_active_taxonomy_is_a_conflict(self):
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant("t-acme", "t-globex")
        assert (e.value.status, e.value.code) == (409, "tenant_conflict")

    def test_nothing_to_go_on_is_refused_not_guessed(self):
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant(None, None)
        assert (e.value.status, e.value.code) == (422, "tenant_required")

    async def test_a_first_discovery_with_a_tenant_is_accepted(self, wired):
        wired()
        c = ctx()
        svc, enqueued = job_service()

        out = await discover(svc, c, tenant_id="t-acme")

        assert out["status"] == "pending" and len(enqueued) == 1
        assert (await svc.get(enqueued[0], c.organization.id)).tenant_id == "t-acme"

    async def test_a_first_discovery_without_a_tenant_is_refused(self, wired):
        wired()
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx())

        assert e.value.status_code == 422 and enqueued == []

    async def test_a_tenant_that_differs_from_the_active_taxonomy_is_refused(self, wired):
        wired(active_tenant="t-acme")
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx(), tenant_id="t-globex")

        assert e.value.status_code == 409 and enqueued == []

    async def test_a_malformed_tenant_is_refused_before_the_job_is_created(self, wired):
        wired()
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx(), tenant_id="../t-globex")

        assert e.value.status_code == 422 and e.value.detail["code"] == "bad_tenant"
        assert enqueued == []
