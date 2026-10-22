r"""The re-label and taxonomy job tables against a real Postgres (the batch-5 gate).

Skipped unless ``AIRWEAVE_NEO_TEST_POSTGRES`` is set to a disposable database that has been
migrated to head (``alembic upgrade head``), for example::

    docker run -d --name gate-pg -e POSTGRES_USER=gate -e POSTGRES_PASSWORD=gate_pw \\
        -e POSTGRES_DB=airweave_scratch -p 55432:5432 postgres:16
    AIRWEAVE_NEO_TEST_POSTGRES=postgresql+asyncpg://gate:gate_pw@localhost:55432/airweave_scratch

The tests insert an invented organization and collection and delete them afterwards.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

URL = os.environ.get("AIRWEAVE_NEO_TEST_POSTGRES")
pytestmark = pytest.mark.skipif(not URL, reason="AIRWEAVE_NEO_TEST_POSTGRES is not set")

from airweave.domains.taxonomy.jobs import records as jr  # noqa: E402
from airweave.domains.taxonomy.relabel import jobs as rj  # noqa: E402
from airweave.domains.taxonomy.relabel import repository as rr  # noqa: E402


@pytest.fixture
async def world():
    engine = create_async_engine(URL)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session():
        async with maker() as db:
            yield db

    org, meta, col_a, col_b = uuid4(), uuid4(), uuid4(), uuid4()
    now = datetime.utcnow()
    async with session() as db:
        await db.execute(
            text(
                "INSERT INTO organization (id, name, created_at, modified_at) "
                "VALUES (:i, 'Acme', :n, :n)"
            ),
            {"i": org, "n": now},
        )
        await db.execute(
            text(
                "INSERT INTO vector_db_deployment_metadata (id, singleton, dense_embedder,"
                " embedding_dimensions, sparse_embedder, created_at, modified_at) VALUES "
                "(:i, true, 'x', 8, 'y', :n, :n) ON CONFLICT DO NOTHING"
            ),
            {"i": meta, "n": now},
        )
        meta_id = (
            await db.execute(text("SELECT id FROM vector_db_deployment_metadata LIMIT 1"))
        ).scalar()
        for cid, rid in (
            (col_a, f"neo-acme-{cid_suffix(col_a)}"),
            (col_b, f"neo-globex-{cid_suffix(col_b)}"),
        ):
            await db.execute(
                text(
                    "INSERT INTO collection (id, name, readable_id, organization_id,"
                    " vector_db_deployment_metadata_id, created_at, modified_at) VALUES "
                    "(:i, 'c', :r, :o, :m, :n, :n)"
                ),
                {"i": cid, "r": rid, "o": org, "m": meta_id, "n": now},
            )
        await db.commit()
    yield {"session": session, "org": org, "a": col_a, "b": col_b}
    async with session() as db:
        await db.execute(text("DELETE FROM organization WHERE id = :i"), {"i": org})  # cascades
        await db.commit()
    await engine.dispose()


def cid_suffix(u):
    return str(u)[:8]


def relabel_job(w, collection="a", **kw):
    base = {
        "id": uuid4(),
        "organization_id": w["org"],
        "collection_id": w[collection],
        "collection_readable_id": "neo-acme",
        "target_version": "v2",
        "spend_cap_usd": 1.0,
    }
    base.update(kw)
    return rj.RelabelJob(**base)


def taxonomy_job(w, collection="a", **kw):
    base = {
        "id": uuid4(),
        "organization_id": w["org"],
        "kind": "discover",
        "collection_id": w[collection],
        "collection_readable_id": "neo-acme",
        "tenant_id": "t-acme",
        "candidate_version": "c1",
        "idempotency_key": "k1",
        "spend_cap_usd": 1.0,
    }
    base.update(kw)
    return jr.TaxonomyJobRecord(**base)


class TestRelabelRepository:
    async def test_create_get_and_one_active_job_per_collection(self, world):
        repo = rr.SqlJobRepository(world["session"])
        first = await repo.create(relabel_job(world, idempotency_key="k1"))
        assert (await repo.get(first.id, world["org"])).target_version == "v2"
        assert await repo.get(first.id, uuid4()) is None  # organization scoped
        with pytest.raises(rj.ActiveJobExists) as e:
            await repo.create(relabel_job(world, idempotency_key="k2"))
        assert e.value.job.id == first.id
        await repo.create(relabel_job(world, collection="b"))  # another collection is independent

    async def test_a_finished_job_frees_the_collection_and_keys_stay_unique(self, world):
        repo = rr.SqlJobRepository(world["session"])
        first = await repo.create(relabel_job(world, idempotency_key="k1"))
        assert await repo.claim(first.id, "w1", 60)
        await repo.finish(first.id, "w1", rj.COMPLETED, None)
        second = await repo.create(relabel_job(world, idempotency_key="k2"))
        assert (await repo.find_by_key(world["a"], "k1")).id == first.id
        # with the collection free again, reusing a key still hits the unique constraint
        assert await repo.claim(second.id, "w1", 60)
        await repo.finish(second.id, "w1", rj.COMPLETED, None)
        with pytest.raises(IntegrityError):
            await repo.create(relabel_job(world, idempotency_key="k1"))

    async def test_claim_is_exclusive_until_the_lease_expires(self, world):
        repo = rr.SqlJobRepository(world["session"])
        job = await repo.create(relabel_job(world))
        assert (await repo.claim(job.id, "w1", 60)).state == "RUNNING"
        assert await repo.claim(job.id, "w2", 60) is None
        assert await repo.claim(job.id, "w1", 60) is not None  # mine again
        assert await repo.save_progress(job.id, "w2", 60, chunks_relabelled=5) is None  # fenced
        saved = await repo.save_progress(
            job.id, "w1", 60, chunks_relabelled=5, checkpoint="[1, null]"
        )
        assert (saved.chunks_relabelled, saved.checkpoint) == (5, "[1, null]")
        async with world["session"]() as db:  # let the lease expire
            await db.execute(
                text("UPDATE taxonomy_relabel_job SET lease_expires_at = :t WHERE id = :i"),
                {"t": datetime.utcnow() - timedelta(seconds=1), "i": job.id},
            )
            await db.commit()
        assert [j.id for j in await repo.resumable()] == [job.id]
        assert (await repo.claim(job.id, "w2", 60)).lease_owner == "w2"
        assert (
            await repo.finish(job.id, "w1", rj.FAILED, "x") is None
        )  # the old owner cannot end it

    async def test_cancel_flags_a_held_job_and_stops_a_free_one(self, world):
        repo = rr.SqlJobRepository(world["session"])
        job = await repo.create(relabel_job(world))
        await repo.claim(job.id, "w1", 60)
        flagged = await repo.request_cancel(job.id)
        assert flagged.cancel_requested and flagged.state == "RUNNING"
        await repo.finish(job.id, "w1", rj.CANCELLED, None)
        free = await repo.create(relabel_job(world, idempotency_key="z"))
        assert (await repo.request_cancel(free.id)).state == "CANCELLED"

    async def test_unannounced_jobs_and_the_sent_flag(self, world):
        repo = rr.SqlJobRepository(world["session"])
        job = await repo.create(relabel_job(world))
        await repo.claim(job.id, "w1", 60)
        await repo.finish(job.id, "w1", rj.COMPLETED, None)
        assert [j.id for j in await repo.unannounced()] == [job.id]
        await repo.mark_event_sent(job.id)
        assert await repo.unannounced() == []


class TestTaxonomyJobRepository:
    async def test_one_active_job_per_collection_and_key_replays(self, world):
        repo = jr.SqlTaxonomyJobRepository(world["session"])
        first = await repo.create(taxonomy_job(world, params={"sample_size": 5}))
        assert (await repo.get(first.id)).params == {"sample_size": 5}
        with pytest.raises(jr.ActiveTaxonomyJob):
            await repo.create(taxonomy_job(world, idempotency_key="k2"))
        assert (await repo.find_by_key(world["a"], "k1")).id == first.id
        assert (await repo.find_active(world["a"])).id == first.id

    async def test_claim_renew_finish_and_cancel_are_fenced(self, world):
        repo = jr.SqlTaxonomyJobRepository(world["session"])
        job = await repo.create(taxonomy_job(world))
        assert await repo.claim(job.id, "w1", 60)
        assert await repo.claim(job.id, "w2", 60) is None
        assert await repo.renew(job.id, "w2", 60, phase="x") is None
        assert (await repo.renew(job.id, "w1", 60, phase="build")).phase == "build"
        assert (await repo.request_cancel(job.id)).cancel_requested
        done = await repo.finish(
            job.id, "w1", jr.COMPLETED, None, result={"files": ["a"]}, spend_used_usd=0.5
        )
        assert (
            done.result == {"files": ["a"]}
            and done.spend_used_usd == 0.5
            and done.lease_owner is None
        )
        await repo.create(taxonomy_job(world, idempotency_key="k2"))  # freed

    async def test_a_free_job_is_cancelled_at_once(self, world):
        repo = jr.SqlTaxonomyJobRepository(world["session"])
        job = await repo.create(taxonomy_job(world))
        assert (await repo.request_cancel(job.id)).state == "CANCELLED"

    async def test_deleting_the_collection_deletes_its_jobs(self, world):
        repo = jr.SqlTaxonomyJobRepository(world["session"])
        job = await repo.create(taxonomy_job(world, collection="b"))
        async with world["session"]() as db:
            await db.execute(text("DELETE FROM collection WHERE id = :i"), {"i": world["b"]})
            await db.commit()
        assert await repo.get(job.id) is None
