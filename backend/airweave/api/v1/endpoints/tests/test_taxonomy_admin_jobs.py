"""The discover and propose endpoints: caps, the tenant a discovery writes to, replays."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from airweave.api.v1.endpoints import taxonomy_admin as ep
from airweave.api.v1.endpoints.tests.test_taxonomy_admin_relabel import CollectionRepo, ctx
from airweave.domains.taxonomy.jobs.records import InMemoryTaxonomyJobRepository
from airweave.domains.taxonomy.jobs.service import (
    TaxonomyJobService,
    neo_collection_id,
    resolve_tenant,
)
from airweave.domains.taxonomy.relabel.service import RelabelRefused

ACME = neo_collection_id("t-acme", "w-default")


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


async def discover(svc, c, readable_id=ACME, **body):
    payload = {"candidate_version": "acme-first", "idempotency_key": "k1", **body}
    return await ep.start_taxonomy_discovery(
        readable_id=readable_id,
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


class TestNeoCollectionId:
    # Produced by neo-platform's deriveCollectionId (packages/backend/security/src/collection-id.ts)
    @pytest.mark.parametrize(
        "tenant, workspace, expected",
        [
            ("t-acme", "w-default", "neo-c018f86d4c910f9ee1e4de67b13a5eeb91c8bd15b53ec4c0"),
            (
                "org_2xYz",
                "8f14e45f-ceea-467f-a0e6-9b7c4a1d2e3f",
                "neo-5b65d40356d2c1d9ab766145220f56ee088faaf3b922455d",
            ),
            ("a:b", "c", "neo-8a37856518dacc57691927d22e62feb58519471ad25e0fc4"),
            ("a", "b:c", "neo-b8d47370810d318bc6f70a5c93e34b860ba61e7638b47e07"),
            ("tenant-é", "ws-😀", "neo-8aba40a889669099176f9ce94be4459b4ac958ddf80313f8"),
        ],
    )
    def test_the_same_id_as_neo(self, tenant, workspace, expected):
        assert neo_collection_id(tenant, workspace) == expected


class TestTenantIsProvenByTheCollection:
    def test_the_pair_that_derives_the_collection_is_accepted(self):
        assert resolve_tenant(ACME, None, "t-acme", "w-default") == "t-acme"

    def test_a_pair_that_names_another_collection_is_refused(self):
        for tenant, workspace in (("t-globex", "w-default"), ("t-acme", "w-other")):
            with pytest.raises(RelabelRefused) as e:
                resolve_tenant(ACME, None, tenant, workspace)
            assert (e.value.status, e.value.code) == (403, "tenant_not_allowed")

    @pytest.mark.parametrize("tenant, workspace", [("t-acme", None), (None, "w-default")])
    def test_half_a_pair_is_refused(self, tenant, workspace):
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant(ACME, None, tenant, workspace)
        assert (e.value.status, e.value.code) == (422, "tenant_scope_incomplete")

    def test_without_a_pair_the_active_file_names_the_tenant(self):
        assert resolve_tenant(ACME, "t-acme", None, None) == "t-acme"

    def test_nothing_to_go_on_is_refused_not_guessed(self):
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant(ACME, None, None, None)
        assert (e.value.status, e.value.code) == (422, "tenant_required")

    def test_an_active_file_that_names_another_tenant_is_a_conflict(self):
        with pytest.raises(RelabelRefused) as e:
            resolve_tenant(ACME, "t-globex", "t-acme", "w-default")
        assert (e.value.status, e.value.code) == (409, "tenant_conflict")

    async def test_discovery_refuses_a_foreign_tenant_before_anything_is_created(self, wired):
        wired()
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx(), tenant_id="t-globex", workspace_id="w-default")

        assert e.value.status_code == 403 and enqueued == []

    async def test_discovery_with_the_collections_own_tenant_and_workspace_is_accepted(
        self, wired
    ):
        wired()
        c = ctx()
        svc, enqueued = job_service()

        out = await discover(svc, c, tenant_id="t-acme", workspace_id="w-default")

        assert out["status"] == "pending" and len(enqueued) == 1
        assert (await svc.get(enqueued[0], c.organization.id)).tenant_id == "t-acme"

    async def test_a_first_discovery_without_tenant_and_workspace_is_refused(self, wired):
        wired()
        svc, enqueued = job_service()

        with pytest.raises(HTTPException) as e:
            await discover(svc, ctx())

        assert e.value.status_code == 422 and enqueued == []

    async def test_a_malformed_tenant_is_refused_before_the_job_is_created(self, wired):
        wired()
        svc, enqueued = job_service()
        readable_id = neo_collection_id("../t-globex", "w-default")

        with pytest.raises(HTTPException) as e:
            await discover(
                svc, ctx(), readable_id, tenant_id="../t-globex", workspace_id="w-default"
            )

        assert e.value.status_code == 422 and e.value.detail["code"] == "bad_tenant"
        assert enqueued == []
