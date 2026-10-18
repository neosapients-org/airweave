"""The discover and propose endpoints: caps, the tenant a discovery writes to, replays."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from airweave.api.v1.endpoints import taxonomy_admin as ep
from airweave.api.v1.endpoints.tests.test_taxonomy_admin_relabel import CollectionRepo, ctx
from airweave.core.config import settings
from airweave.domains.taxonomy.jobs.records import InMemoryTaxonomyJobRepository
from airweave.domains.taxonomy.jobs.service import TaxonomyJobService, resolve_tenant
from airweave.domains.taxonomy.relabel.service import RelabelRefused


def job_service(cap=5.0):
    enqueued = []

    async def enqueue(job):
        enqueued.append(job.id)

    repo = InMemoryTaxonomyJobRepository()
    return TaxonomyJobService(repo, enqueue, discover_cap_usd=cap, evolution_cap_usd=cap), enqueued


class Resolver:
    def __init__(self, tenant=None):
        self.tenant = tenant

    async def resolve(self, readable_id, fresh=False):
        return SimpleNamespace(tenant_id=self.tenant) if self.tenant else None


@pytest.fixture
def wired(monkeypatch):
    def wire(tenant_bindings=None, active_tenant=None):
        monkeypatch.setattr(settings, "TAXONOMY_TENANT_BY_ORGANIZATION", tenant_bindings or {})
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


class TestTenantIsNotTheCallers:
    def test_the_organizations_binding_wins_and_a_matching_request_is_accepted(self):
        org = uuid4()
        assert resolve_tenant(org, {str(org): "t-acme"}, None, "t-acme") == "t-acme"
        assert resolve_tenant(org, {str(org): "t-acme"}, None, None) == "t-acme"

    def test_the_active_file_names_the_tenant_when_no_binding_is_configured(self):
        assert resolve_tenant(uuid4(), {}, "t-acme", None) == "t-acme"

    def test_a_tenant_the_caller_names_must_be_the_resolved_one(self):
        org = uuid4()
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant(org, {str(org): "t-acme"}, None, "t-globex")
        assert (e.value.status, e.value.code) == (403, "tenant_not_allowed")

    def test_nothing_to_go_on_is_refused_not_guessed(self):
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant(uuid4(), {"someone-else": "t-globex"}, None, "t-globex")
        assert (e.value.status, e.value.code) == (422, "tenant_not_configured")

    def test_a_binding_that_contradicts_the_active_file_is_a_conflict(self):
        org = uuid4()
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant(org, {str(org): "t-acme"}, "t-globex", None)
        assert e.value.status == 409

    async def test_discovery_refuses_a_foreign_tenant_before_anything_is_created(self, wired):
        org = uuid4()
        wired({str(org): "t-acme"})
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx(org=org), tenant_id="t-globex")

        assert e.value.status_code == 403 and enqueued == []

    async def test_discovery_without_a_tenant_in_the_body_uses_the_bound_one(self, wired):
        org = uuid4()
        wired({str(org): "t-acme"})
        svc, enqueued = job_service()

        out = await discover(svc, ctx(org=org))

        assert out["status"] == "running" and len(enqueued) == 1
        assert (await svc.get(enqueued[0], org)) is not None

    async def test_an_organization_with_no_binding_and_no_active_file_cannot_discover(self, wired):
        wired({})
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx(), tenant_id="t-anyone")

        assert e.value.status_code == 422 and enqueued == []

    async def test_a_malformed_tenant_in_the_binding_is_refused_before_the_job_is_created(
        self, wired
    ):
        org = uuid4()
        wired({str(org): "../t-globex"})
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx(org=org))

        assert e.value.status_code == 422 and e.value.detail["code"] == "bad_tenant"
        assert enqueued == []
