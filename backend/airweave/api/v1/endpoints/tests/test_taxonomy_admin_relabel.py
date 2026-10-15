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
