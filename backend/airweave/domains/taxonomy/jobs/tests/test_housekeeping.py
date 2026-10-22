"""Recovering jobs nothing runs, failing abandoned ones, and expiring retained caches."""

import os
import time
from datetime import timedelta
from uuid import UUID

from airweave.domains.taxonomy.jobs import housekeeping
from airweave.domains.taxonomy.jobs.housekeeping import (
    delete_tenant_cache,
    purge_expired_caches,
    purge_finished_work,
    reconcile_unattended,
    tenant_cache_dir,
)
from airweave.domains.taxonomy.jobs.records import (
    FAILED,
    PENDING,
    RUNNING,
    InMemoryTaxonomyJobRepository,
    utcnow,
)
from airweave.domains.taxonomy.jobs.tests.helpers import job

GRACE, MAX_AGE = 120, 24 * 3600


def seeded(*, state=PENDING, age=0, lease_in=None, collection_int=11):
    repo = InMemoryTaxonomyJobRepository()
    j = job(collection_id=UUID(int=collection_int))
    j.state = state
    j.created_at = utcnow() - timedelta(seconds=age)
    if lease_in is not None:
        j.lease_owner = "worker-1"
        j.lease_expires_at = utcnow() + timedelta(seconds=lease_in)
    repo.jobs[j.id] = j
    return repo, j


async def reconcile(repo):
    started, cleaned = [], []

    async def dispatch(j):
        started.append(j.id)

    out = await reconcile_unattended(
        repo, dispatch, cleaned.append, grace_seconds=GRACE, max_age_seconds=MAX_AGE
    )
    return out, started, cleaned


class TestReconcile:
    async def test_a_pending_job_whose_dispatch_never_arrived_is_started(self):
        repo, j = seeded(age=600)
        out, started, _ = await reconcile(repo)
        assert started == [j.id] and out == {"started": 1, "abandoned": 0}

    async def test_a_job_still_inside_the_grace_period_is_left_to_its_own_dispatch(self):
        repo, _ = seeded(age=30)
        out, started, _ = await reconcile(repo)
        assert started == [] and out["started"] == 0

    async def test_a_running_job_with_a_live_lease_is_left_alone(self):
        repo, _ = seeded(state=RUNNING, age=900, lease_in=200)
        _, started, _ = await reconcile(repo)
        assert started == []

    async def test_a_running_job_whose_lease_lapsed_is_started_again(self):
        repo, j = seeded(state=RUNNING, age=900, lease_in=-10)
        _, started, _ = await reconcile(repo)
        assert started == [j.id]

    async def test_a_job_unattended_for_too_long_is_failed_and_its_work_removed(self):
        repo, j = seeded(state=RUNNING, age=MAX_AGE + 60, lease_in=-10)
        out, started, cleaned = await reconcile(repo)
        assert started == [] and cleaned == [j.id] and out["abandoned"] == 1
        done = await repo.get(j.id)
        assert done.state == FAILED and "start it again" in done.error
        assert done.lease_owner is None

    async def test_a_finished_job_is_never_touched(self):
        repo, j = seeded(age=MAX_AGE * 2)
        repo.jobs[j.id].state = "COMPLETED"
        _, started, cleaned = await reconcile(repo)
        assert started == [] and cleaned == []

    async def test_an_unreachable_temporal_does_not_stop_the_round_and_is_tried_again(self):
        repo, j = seeded(age=600)
        calls = []

        async def down(job_):
            calls.append(job_.id)
            raise ConnectionError("temporal is down")

        out = await reconcile_unattended(
            repo, down, lambda _: None, grace_seconds=GRACE, max_age_seconds=MAX_AGE
        )
        assert out["started"] == 0 and calls == [j.id]
        assert (await repo.get(j.id)).state == PENDING  # still there for the next round

    async def test_the_workflow_start_is_idempotent_so_a_repeated_round_is_safe(self):
        repo, j = seeded(age=600)
        _, first, _ = await reconcile(repo)
        _, second, _ = await reconcile(repo)
        assert (
            first == second == [j.id]
        )  # same id each time: Temporal's USE_EXISTING makes it a no-op


class TestAbandonedWork:
    async def test_a_failed_removal_is_logged_as_an_error_and_the_round_goes_on(self, monkeypatch):
        repo, j = seeded(state=RUNNING, age=MAX_AGE + 60, lease_in=-10)
        errors = []
        monkeypatch.setattr(housekeeping.logger, "error", errors.append)

        def refuse(_):
            raise PermissionError("read-only file system")

        out = await reconcile_unattended(
            repo, lambda _: None, refuse, grace_seconds=GRACE, max_age_seconds=MAX_AGE
        )
        assert out["abandoned"] == 1 and (await repo.get(j.id)).state == FAILED
        assert any("could not remove the work" in m for m in errors)

    async def test_the_next_round_removes_the_work_of_every_job_that_is_no_longer_active(
        self, tmp_path
    ):
        repo, finished = seeded(state=FAILED)
        _, running = seeded(state=RUNNING, lease_in=200)
        repo.jobs[running.id] = running
        for j in (finished, running):
            (tmp_path / str(j.id)).mkdir()
            (tmp_path / str(j.id) / "sample.jsonl").write_text("customer text")
        gone = UUID(int=99)  # a job whose row no longer exists
        (tmp_path / str(gone)).mkdir()
        (tmp_path / "not-a-job").mkdir()

        removed = await purge_finished_work(repo, tmp_path)

        assert sorted(removed) == sorted([str(finished.id), str(gone)])
        assert (tmp_path / str(running.id)).is_dir() and (tmp_path / "not-a-job").is_dir()

    async def test_a_folder_that_cannot_be_removed_stays_for_the_next_round(
        self, tmp_path, monkeypatch
    ):
        repo, finished = seeded(state=FAILED)
        (tmp_path / str(finished.id)).mkdir()
        errors = []
        monkeypatch.setattr(housekeeping.logger, "error", errors.append)

        def refuse(_path):
            raise OSError("device busy")

        monkeypatch.setattr(housekeeping.shutil, "rmtree", refuse)
        assert await purge_finished_work(repo, tmp_path) == []
        assert (tmp_path / str(finished.id)).is_dir()
        assert any("could not remove the work" in m for m in errors)

        monkeypatch.undo()
        assert await purge_finished_work(repo, tmp_path) == [str(finished.id)]


class TestCaches:
    def make(self, root, tenant, age_days):
        folder = tenant_cache_dir(root, tenant)
        (folder / "llm_cache").mkdir(parents=True)
        f = folder / "llm_cache" / "x.json"
        f.write_text("customer text")
        old = time.time() - age_days * 86400
        for p in (f, f.parent, folder):
            os.utime(p, (old, old))
        return folder

    def test_the_folder_name_is_a_hash_not_the_tenant_id(self, tmp_path):
        assert "t-acme" not in str(tenant_cache_dir(tmp_path, "t-acme"))
        assert tenant_cache_dir(tmp_path, "t-acme") != tenant_cache_dir(tmp_path, "t-globex")

    def test_an_unused_cache_past_retention_is_removed_and_a_fresh_one_kept(self, tmp_path):
        old = self.make(tmp_path, "t-acme", age_days=45)
        fresh = self.make(tmp_path, "t-globex", age_days=2)

        removed = purge_expired_caches(tmp_path, retention_days=30)

        assert removed == [old.name] and not old.exists() and fresh.exists()

    def test_a_failed_deletion_is_not_reported_as_removed_and_is_tried_again(
        self, tmp_path, monkeypatch
    ):
        old = self.make(tmp_path, "t-acme", age_days=45)

        def refuse(_path):
            raise PermissionError("read-only")

        monkeypatch.setattr(housekeeping.shutil, "rmtree", refuse)
        assert purge_expired_caches(tmp_path, retention_days=30) == [] and old.exists()

        monkeypatch.undo()
        assert purge_expired_caches(tmp_path, retention_days=30) == [old.name]

    def test_a_folder_is_as_old_as_its_newest_file(self, tmp_path):
        folder = self.make(tmp_path, "t-acme", age_days=45)
        (folder / "llm_cache" / "new.json").write_text("used today")

        assert purge_expired_caches(tmp_path, retention_days=30) == []
        assert folder.exists()

    def test_zero_keeps_everything_for_ever(self, tmp_path):
        folder = self.make(tmp_path, "t-acme", age_days=4000)
        assert purge_expired_caches(tmp_path, retention_days=0) == [] and folder.exists()

    def test_a_missing_directory_is_fine(self, tmp_path):
        assert purge_expired_caches(tmp_path / "nope", retention_days=30) == []

    def test_deleting_a_tenant_removes_its_folder_at_once_and_only_its_own(self, tmp_path):
        mine = self.make(tmp_path, "t-acme", age_days=0)
        other = self.make(tmp_path, "t-globex", age_days=0)

        assert delete_tenant_cache(tmp_path, "t-acme") is True
        assert not mine.exists() and other.exists()
        assert delete_tenant_cache(tmp_path, "t-acme") is False

    def test_the_command_line_deletes_one_tenants_cache(self, tmp_path, monkeypatch, capsys):
        from airweave.core.config import settings

        folder = self.make(tmp_path, "t-acme", age_days=0)
        monkeypatch.setattr(settings, "TAXONOMY_JOBS_CACHE_DIR", str(tmp_path))

        assert housekeeping._main(["delete-tenant", "t-acme"]) == 0
        assert not folder.exists() and "removed" in capsys.readouterr().out
        assert housekeeping._main(["nonsense"]) == 2
