"""The re-label admin endpoints, called directly (bypassing FastAPI DI) with fakes."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from airweave.api.context import ApiContext
from airweave.api.v1.endpoints import taxonomy_admin as ep
from airweave.core.logging import logger
from airweave.core.shared_models import AuthMethod, FeatureFlag
from airweave.domains.taxonomy.relabel.jobs import InMemoryJobRepository
from airweave.domains.taxonomy.relabel.service import RelabelService
from airweave.domains.taxonomy.relabel.tests.helpers import (
    COLLECTION,
    NEW,
    OLD,
    FakeChunkStore,
    chunk,
)


def ctx(features=(FeatureFlag.API_KEY_ADMIN_SYNC,), auth=AuthMethod.API_KEY, org=None):
    from airweave.schemas.organization import Organization

    now = datetime.now(timezone.utc)
    organization = Organization(
        id=str(org or uuid4()),
        name="Acme",
        created_at=now,
        modified_at=now,
        enabled_features=list(features),
    )
    return ApiContext(
        request_id="r",
        organization=organization,
        auth_method=auth,
        auth_metadata={},
        logger=logger.with_context(request_id="r"),
    )


class CollectionRepo:
    def __init__(self, found=True):
        self.found = found

    async def get_by_readable_id(self, db, readable_id, ctx):
        return SimpleNamespace(id=COLLECTION, readable_id=readable_id) if self.found else None


def service(cap=10.0, store=None):
    store = store or FakeChunkStore([chunk(i, status="ASSIGNED", version=OLD) for i in range(3)])
    return RelabelService(
        InMemoryJobRepository(), lambda: store, configured_cap_usd=cap, usd_per_thousand=0.006
    )


async def start(svc, c, body=None, found=True, key=None):
    return await ep.start_relabel(
        readable_id="neo-acme",
        body=body or ep.RelabelRequest(taxonomy_version=NEW),
        idempotency_key=key,
        db=AsyncMock(),
        ctx=c,
        collection_repo=CollectionRepo(found),
        service=svc,
    )


class TestStart:
    async def test_accepts_and_returns_the_contract_fields(self):
        c = ctx()
        out = await start(service(), c)
        assert (
            UUID(out["relabel_job_id"]) and out["status"] == "running" and out["state"] == "PENDING"
        )
        assert out["taxonomy_version"] == NEW and out["replayed"] is False
        assert out["chunks_total"] == 3 and out["error"] is None

    async def test_header_idempotency_key_makes_a_replay_return_the_job(self):
        svc, c = service(), ctx()
        first = await start(svc, c, key="mig-7")
        again = await start(svc, c, key="mig-7")
        assert again["relabel_job_id"] == first["relabel_job_id"] and again["replayed"] is True

    async def test_body_key_wins_over_the_header(self):
        svc, c = service(), ctx()
        body = ep.RelabelRequest(taxonomy_version=NEW, idempotency_key="body-key")
        first = await start(svc, c, body, key="header-key")
        job = await svc.get(UUID(first["relabel_job_id"]), c.organization.id)
        assert job.idempotency_key == "body-key"

    async def test_unknown_collection_is_404(self):
        with pytest.raises(HTTPException) as e:
            await start(service(), ctx(), found=False)
        assert e.value.status_code == 404

    async def test_spend_cap_refusal_is_422_with_a_code(self):
        with pytest.raises(HTTPException) as e:
            await start(service(cap=0.0), ctx())
        assert e.value.status_code == 422 and e.value.detail["code"] == "spend_cap_not_approved"

    async def test_conflict_is_409_naming_the_running_job(self):
        svc, c = service(), ctx()
        first = await start(svc, c)
        with pytest.raises(HTTPException) as e:
            await start(svc, c, ep.RelabelRequest(taxonomy_version="acme-2027.01"))
        assert (
            e.value.status_code == 409
            and e.value.detail["relabel_job_id"] == first["relabel_job_id"]
        )

    def test_unknown_body_fields_are_rejected(self):
        with pytest.raises(ValueError):
            ep.RelabelRequest(taxonomy_version=NEW, surprise=1)


class TestPermission:
    @pytest.mark.parametrize("call", ["start", "get", "cancel"])
    async def test_api_key_without_the_admin_feature_is_refused(self, call):
        c, svc = ctx(features=()), service()
        with pytest.raises(HTTPException) as e:
            if call == "start":
                await start(svc, c)
            elif call == "get":
                await ep.get_relabel_job(uuid4(), ctx=c, service=svc)
            else:
                await ep.cancel_relabel_job(uuid4(), ctx=c, service=svc)
        assert e.value.status_code == 403

    async def test_a_plain_user_is_refused(self):
        c = ctx(features=(), auth=AuthMethod.SYSTEM)
        with pytest.raises(HTTPException) as e:
            await start(service(), c)
        assert e.value.status_code == 403

    async def test_an_admin_user_is_allowed(self):
        c = ctx(features=(), auth=AuthMethod.AUTH0)
        object.__setattr__(
            c, "user", SimpleNamespace(is_admin=True, is_superuser=False, email="a@x")
        )
        out = await start(service(), c)
        assert out["state"] == "PENDING"


class TestGetAndCancel:
    async def test_get_reports_the_contract_shape(self):
        svc, c = service(), ctx()
        job = await start(svc, c)
        out = await ep.get_relabel_job(UUID(job["relabel_job_id"]), ctx=c, service=svc)
        assert {"status", "chunks_relabelled", "chunks_total", "error", "state"} <= set(out)

    @pytest.mark.parametrize(
        "state, status",
        [
            ("RUNNING", "running"),
            ("PENDING", "running"),
            ("COMPLETED", "completed"),
            ("FAILED", "failed"),
            ("CANCELLED", "cancelled"),
        ],
    )
    def test_status_words(self, state, status):
        from airweave.domains.taxonomy.relabel.tests.helpers import new_job

        job = new_job(state=state, chunks_relabelled=3, chunks_stamped=2)
        view = ep.job_view(job)
        assert view["status"] == status and view["state"] == state
        assert view["chunks_relabelled"] == 5 and view["chunks_classified"] == 3

    async def test_another_organizations_job_is_not_found(self):
        svc = service()
        job = await start(svc, ctx())
        other = ctx()
        for call in (ep.get_relabel_job, ep.cancel_relabel_job):
            with pytest.raises(HTTPException) as e:
                await call(UUID(job["relabel_job_id"]), ctx=other, service=svc)
            assert e.value.status_code == 404

    async def test_cancel_stops_an_unclaimed_job(self):
        svc, c = service(), ctx()
        job = await start(svc, c)
        out = await ep.cancel_relabel_job(UUID(job["relabel_job_id"]), ctx=c, service=svc)
        assert out["state"] == "CANCELLED" and out["status"] == "cancelled"

    async def test_unknown_job_is_404(self):
        with pytest.raises(HTTPException) as e:
            await ep.get_relabel_job(uuid4(), ctx=ctx(), service=service())
        assert e.value.status_code == 404


def test_503_while_the_taxonomy_step_is_off(monkeypatch):
    from airweave.core import container as container_mod

    monkeypatch.setattr(
        container_mod, "container", SimpleNamespace(relabel_service=None), raising=False
    )
    with pytest.raises(HTTPException) as e:
        ep.get_relabel_service()
    assert e.value.status_code == 503


def test_routes_are_mounted_under_admin():
    from airweave.api.v1.api import api_router

    paths = {(m, r.path) for r in api_router.routes for m in getattr(r, "methods", ())}
    assert ("POST", "/admin/collections/{readable_id}/relabel") in paths
    assert ("GET", "/admin/relabel-jobs/{relabel_job_id}") in paths
    assert ("POST", "/admin/relabel-jobs/{relabel_job_id}/cancel") in paths


class TestStatisticsEndpoint:
    def world(self):
        return FakeChunkStore(
            [
                chunk(0, doc="a", status="ASSIGNED", version=OLD),
                chunk(1, doc="a", status="UNRESOLVED", version=OLD, unresolved=1),
                chunk(0, doc="b", status="ASSIGNED", version=NEW),
                chunk(0, doc="c"),
            ]
        )

    async def call(self, svc, c=None, target=NEW, found=True):
        return await ep.taxonomy_statistics(
            readable_id="neo-acme",
            target_version=target,
            db=AsyncMock(),
            ctx=c or ctx(),
            collection_repo=CollectionRepo(found),
            service=svc,
        )

    async def test_returns_the_ports_fields_and_the_airweave_collection_id(self):
        out = await self.call(service(store=self.world()))
        assert out["collectionId"] == str(COLLECTION)
        assert (out["total"], out["otherVersion"], out["documentsAtVersion"]) == (4, 3, 1)
        assert out["incrementalBacklog"] == 2 and out["labelled"] == 2
        assert {
            "errorChunks",
            "pieces",
            "unresolvedPieces",
            "notLabelledChunks",
            "notLabelledDocuments",
        } <= set(out)

    async def test_callable_with_an_empty_target(self):
        out = await self.call(service(store=self.world()), target="")
        assert out["documentsAtVersion"] == 0 and out["notLabelledChunks"] == 1

    async def test_needs_the_admin_permission(self):
        with pytest.raises(HTTPException) as e:
            await self.call(service(store=self.world()), c=ctx(features=()))
        assert e.value.status_code == 403

    async def test_unknown_collection_is_404(self):
        with pytest.raises(HTTPException) as e:
            await self.call(service(store=self.world()), found=False)
        assert e.value.status_code == 404

    async def test_malformed_target_is_422(self):
        with pytest.raises(HTTPException) as e:
            await self.call(service(store=self.world()), target="../x")
        assert e.value.status_code == 422

    async def test_store_is_closed_and_nothing_is_written(self):
        store = self.world()
        await self.call(service(store=store))
        assert store.closed and store.updates == []

    async def test_vespa_failure_is_a_502_not_a_wrong_answer(self):
        class Down(FakeChunkStore):
            async def scroll(self, *a, **k):
                raise RuntimeError("vespa down")

        with pytest.raises(HTTPException) as e:
            await self.call(service(store=Down()))
        assert e.value.status_code == 502

    def test_route_is_mounted(self):
        from airweave.api.v1.api import api_router

        paths = {(m, r.path) for r in api_router.routes for m in getattr(r, "methods", ())}
        assert ("GET", "/admin/collections/{readable_id}/taxonomy-statistics") in paths


class TestUnresolvedEndpoint:
    def world(self):
        return FakeChunkStore(
            [chunk(i, doc="a", status="UNRESOLVED", version=NEW, unresolved=1) for i in range(3)]
            + [chunk(0, doc="b", status="ASSIGNED", version=NEW)]
        )

    async def call(self, svc, c=None, limit=100, cursor=None, version=None, found=True):
        return await ep.list_unresolved(
            readable_id="neo-acme",
            limit=limit,
            cursor=cursor,
            taxonomy_version=version,
            db=AsyncMock(),
            ctx=c or ctx(),
            collection_repo=CollectionRepo(found),
            service=svc,
        )

    async def test_returns_chunks_and_a_cursor_field(self):
        out = await self.call(service(store=self.world()))
        assert len(out["chunks"]) == 3 and out["next_cursor"] is None
        assert {"text", "document_id", "taxonomy_version", "positions"} <= set(out["chunks"][0])

    async def test_pages_with_the_cursor(self):
        svc = service(store=self.world())
        seen, cursor = [], None
        for _ in range(10):
            out = await self.call(svc, limit=1, cursor=cursor)
            seen += [c["chunk_id"] for c in out["chunks"]]
            cursor = out["next_cursor"]
            if cursor is None:
                break
        assert len(seen) == 3 and len(set(seen)) == 3

    async def test_needs_the_admin_permission(self):
        with pytest.raises(HTTPException) as e:
            await self.call(service(store=self.world()), c=ctx(features=()))
        assert e.value.status_code == 403

    async def test_unknown_collection_is_404(self):
        with pytest.raises(HTTPException) as e:
            await self.call(service(store=self.world()), found=False)
        assert e.value.status_code == 404

    @pytest.mark.parametrize(
        "kw", [{"limit": 0}, {"limit": 10_000}, {"cursor": "junk"}, {"version": "../x"}]
    )
    async def test_bad_input_is_422(self, kw):
        with pytest.raises(HTTPException) as e:
            await self.call(service(store=self.world()), **kw)
        assert e.value.status_code == 422

    async def test_read_only(self):
        store = self.world()
        await self.call(service(store=store))
        assert store.updates == [] and store.closed

    def test_route_is_mounted(self):
        from airweave.api.v1.api import api_router

        paths = {(m, r.path) for r in api_router.routes for m in getattr(r, "methods", ())}
        assert ("GET", "/admin/collections/{readable_id}/unresolved") in paths


class TestTaxonomyJobEndpoints:
    def svc(self, cap=5.0):
        from airweave.domains.taxonomy.jobs.records import InMemoryTaxonomyJobRepository
        from airweave.domains.taxonomy.jobs.service import TaxonomyJobService

        async def enqueue(job):
            return None

        return TaxonomyJobService(
            InMemoryTaxonomyJobRepository(), enqueue, discover_cap_usd=cap, evolution_cap_usd=cap
        )

    def resolver(self, result="ok"):
        from airweave.domains.taxonomy.relabel.tests.helpers import FakeResolver

        return FakeResolver(None if result == "ok" else result)

    def container(self, monkeypatch, resolver):
        from airweave.core import container as container_mod

        monkeypatch.setattr(
            container_mod,
            "container",
            SimpleNamespace(taxonomy_processor=SimpleNamespace(resolver=resolver)),
            raising=False,
        )

    async def discover(self, svc, c=None, found=True, **kw):
        body = ep.DiscoveryRequest(
            **{
                "candidate_version": "acme-first",
                "idempotency_key": "k1",
                "tenant_id": "t-acme",
                **kw,
            }
        )
        return await ep.start_taxonomy_discovery(
            readable_id="neo-acme",
            body=body,
            db=AsyncMock(),
            ctx=c or ctx(),
            collection_repo=CollectionRepo(found),
            service=svc,
        )

    async def propose(self, svc, c=None, **kw):
        body = ep.ProposalRequest(**{"candidate_version": "acme-v2", "idempotency_key": "p1", **kw})
        return await ep.start_taxonomy_proposal(
            readable_id="neo-acme",
            body=body,
            db=AsyncMock(),
            ctx=c or ctx(),
            collection_repo=CollectionRepo(True),
            service=svc,
        )

    async def test_discover_accepts_and_replays(self):
        svc = self.svc()
        first = await self.discover(svc)
        again = await self.discover(svc)
        assert (
            first["kind"] == "discover"
            and first["state"] == "PENDING"
            and first["replayed"] is False
        )
        assert again["job_id"] == first["job_id"] and again["replayed"] is True

    async def test_discover_needs_a_cap(self):
        with pytest.raises(HTTPException) as e:
            await self.discover(self.svc(cap=0.0))
        assert e.value.status_code == 422 and e.value.detail["code"] == "spend_cap_not_approved"

    async def test_propose_takes_the_tenant_from_the_active_file(self, monkeypatch):
        self.container(monkeypatch, self.resolver())
        out = await self.propose(self.svc())
        assert out["kind"] == "propose" and out["state"] == "PENDING"

    async def test_propose_without_an_active_taxonomy_is_409(self, monkeypatch):
        self.container(monkeypatch, self.resolver("none"))
        with pytest.raises(HTTPException) as e:
            await self.propose(self.svc())
        assert e.value.status_code == 409 and e.value.detail["code"] == "no_active_taxonomy"

    async def test_propose_with_an_unusable_taxonomy_is_409(self, monkeypatch):
        from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError

        self.container(monkeypatch, self.resolver(TaxonomyResolutionError("checksum mismatch")))
        with pytest.raises(HTTPException) as e:
            await self.propose(self.svc())
        assert e.value.status_code == 409 and "checksum mismatch" in e.value.detail["message"]

    @pytest.mark.parametrize("call", ["discover", "propose", "get", "cancel"])
    async def test_admin_permission_is_required(self, call, monkeypatch):
        self.container(monkeypatch, self.resolver())
        c, svc = ctx(features=()), self.svc()
        with pytest.raises(HTTPException) as e:
            if call == "discover":
                await self.discover(svc, c)
            elif call == "propose":
                await self.propose(svc, c)
            elif call == "get":
                await ep.get_taxonomy_job(uuid4(), ctx=c, service=svc)
            else:
                await ep.cancel_taxonomy_job(uuid4(), ctx=c, service=svc)
        assert e.value.status_code == 403

    async def test_unknown_collection_is_404(self):
        with pytest.raises(HTTPException) as e:
            await self.discover(self.svc(), found=False)
        assert e.value.status_code == 404

    async def test_get_and_cancel_are_organization_scoped(self):
        svc = self.svc()
        job = await self.discover(svc)
        for call in (ep.get_taxonomy_job, ep.cancel_taxonomy_job):
            with pytest.raises(HTTPException) as e:
                await call(UUID(job["job_id"]), ctx=ctx(), service=svc)
            assert e.value.status_code == 404

    async def test_cancel_an_unclaimed_job(self):
        svc, c = self.svc(), ctx()
        job = await self.discover(svc, c)
        out = await ep.cancel_taxonomy_job(UUID(job["job_id"]), ctx=c, service=svc)
        assert out["state"] == "CANCELLED" and out["status"] == "cancelled"

    def test_unknown_body_fields_are_refused(self):
        with pytest.raises(ValueError):
            ep.DiscoveryRequest(
                candidate_version="a", idempotency_key="k", tenant_id="t", surprise=1
            )
        with pytest.raises(ValueError):
            ep.ProposalRequest(candidate_version="a", idempotency_key="k", tenant_id="t")

    def test_503_while_the_jobs_are_off(self, monkeypatch):
        from airweave.core import container as container_mod

        monkeypatch.setattr(
            container_mod, "container", SimpleNamespace(taxonomy_job_service=None), raising=False
        )
        with pytest.raises(HTTPException) as e:
            ep.get_job_service()
        assert e.value.status_code == 503

    def test_routes_are_mounted(self):
        from airweave.api.v1.api import api_router

        paths = {(m, r.path) for r in api_router.routes for m in getattr(r, "methods", ())}
        assert ("POST", "/admin/collections/{readable_id}/taxonomy-discovery") in paths
        assert ("POST", "/admin/collections/{readable_id}/taxonomy-evolution/propose") in paths
        assert ("GET", "/admin/taxonomy-jobs/{job_id}") in paths
        assert ("POST", "/admin/taxonomy-jobs/{job_id}/cancel") in paths
