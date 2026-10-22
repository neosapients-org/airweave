"""What keeps the jobs from stranding work or data: recovery of unattended jobs and cache expiry.

Two things are written outside a job's own lease and have no owner once a worker is gone:

* a job row that is still active but that nothing runs (the request's dispatch never reached
  Temporal, the workflow ended after its retries or its ``busy`` waits, the worker died). Its
  collection stays blocked until something starts it again, so the jobs worker's reconciler does
  that, and fails a job that has been unattended for longer than any build takes;
* a job's scratch folder (its sample of customer text), which its run removes when it ends. One
  left behind (an abandoned job, a removal that failed, a worker that died) is removed by the
  next round, and one that cannot be removed is logged as an error and tried again;
* the persistent per-tenant caches (``TAXONOMY_JOBS_CACHE_DIR``), which hold customer text and the
  model's answers about it. They are removed after ``limits.JOBS_CACHE_RETENTION_DAYS`` without
  use, and a tenant's can be deleted at once with ``delete_tenant_cache`` (or the command line:
  ``python -m airweave.domains.taxonomy.jobs.housekeeping delete-tenant <tenant id>``).
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import UUID

from airweave.core.logging import logger
from airweave.domains.taxonomy.jobs.records import ACTIVE, TaxonomyJobRepository, utcnow

ABANDONED = "the job was not run for too long and was abandoned; start it again"


def tenant_cache_dir(cache_dir: Path, tenant_id: str) -> Path:
    """The folder one tenant's retained caches live in (a hash: no tenant id on disk)."""
    return cache_dir / hashlib.sha256(tenant_id.encode()).hexdigest()[:32]


def delete_tenant_cache(cache_dir: str | Path, tenant_id: str) -> bool:
    """Remove a tenant's retained caches now (the tenant's data is being deleted).

    Returns:
        True when a folder was removed.
    """
    folder = tenant_cache_dir(Path(cache_dir), tenant_id)
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    logger.info(f"[TaxonomyJob] removed the retained caches of a tenant ({folder.name})")
    return True


def _newest_use(folder: Path) -> float:
    newest = folder.stat().st_mtime
    for path in folder.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def purge_expired_caches(
    cache_dir: str | Path, retention_days: int, now: Callable[[], float] = time.time
) -> List[str]:
    """Remove the tenant folders nothing has touched for ``retention_days`` (0: keep for ever).

    Returns:
        The names of the folders removed (one that could not be removed is logged and retried).
    """
    root = Path(cache_dir)
    if retention_days <= 0 or not root.is_dir():
        return []
    cutoff = now() - retention_days * 86400
    removed: List[str] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if _newest_use(folder) < cutoff:
            try:
                shutil.rmtree(folder)
            except OSError as e:
                # Customer text stays on disk: say so, and leave the folder for the next pass.
                logger.error(f"[TaxonomyJob] could not remove the expired cache {folder.name}: {e}")
                continue
            removed.append(folder.name)
    if removed:
        logger.info(
            f"[TaxonomyJob] removed {len(removed)} retained caches unused for {retention_days} days"
        )
    return removed


Dispatch = Callable[[Any], Awaitable[None]]


async def reconcile_unattended(
    repo: TaxonomyJobRepository,
    dispatch: Dispatch,
    cleanup: Callable[[Any], None],
    *,
    grace_seconds: int,
    max_age_seconds: int,
    now: Callable[[], datetime] = utcnow,
    limit: int = 50,
) -> Dict[str, int]:
    """Start a workflow for each active job nothing runs; fail the ones that waited too long.

    A job is *unattended* when it is active and no live lease holds it. ``grace_seconds`` keeps
    this from racing a dispatch that is still on its way. A started workflow that is already
    running is left alone (the start is ``USE_EXISTING``), so this is safe to repeat.
    """
    moment = now()
    started = abandoned = 0
    for job in await repo.list_unattended(moment - timedelta(seconds=grace_seconds), limit):
        if job.created_at <= moment - timedelta(seconds=max_age_seconds):
            failed = await repo.abandon(
                job.id, ABANDONED, moment - timedelta(seconds=max_age_seconds)
            )
            if failed is not None:
                abandoned += 1
                try:
                    cleanup(job.id)
                except OSError as e:
                    # Customer text stays on disk: said as an error, and the folder is removed by
                    # a later round's purge_finished_work.
                    logger.error(
                        f"[TaxonomyJob] could not remove the work of abandoned job {job.id}: {e}"
                    )
            continue
        try:
            await dispatch(job)
            started += 1
        except Exception as e:  # noqa: BLE001 - Temporal is unreachable; the next round retries
            logger.warning(f"[TaxonomyJob] could not start job {job.id}: {type(e).__name__}: {e}")
    return {"started": started, "abandoned": abandoned}


async def purge_finished_work(
    repo: TaxonomyJobRepository, work_dir: Optional[str | Path]
) -> List[str]:
    """Remove the scratch folder of every job that is no longer active, or no longer exists.

    A job's run removes its own folder when it ends; this removes what is left. A folder that
    cannot be removed is logged as an error and stays for the next round.

    Returns:
        The names (job ids) of the folders removed.
    """
    root = Path(work_dir) if work_dir else None
    if root is None or not root.is_dir():
        return []
    removed: List[str] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            job_id = UUID(folder.name)
        except ValueError:
            continue  # not a job's folder
        job = await repo.get(job_id)
        if job is not None and job.state in ACTIVE:
            continue
        try:
            await asyncio.to_thread(shutil.rmtree, folder)
        except OSError as e:
            logger.error(f"[TaxonomyJob] could not remove the work of job {folder.name}: {e}")
            continue
        removed.append(folder.name)
    if removed:
        logger.info(f"[TaxonomyJob] removed the work left by {len(removed)} finished jobs")
    return removed


async def housekeeping_loop(
    stop: asyncio.Event,
    repo: TaxonomyJobRepository,
    dispatch: Dispatch,
    cleanup: Callable[[Any], None],
    *,
    interval_seconds: float,
    grace_seconds: int,
    max_age_seconds: int,
    cache_dir: Optional[str],
    retention_days: int,
    work_dir: Optional[str] = None,
) -> None:
    """Run the tasks every ``interval_seconds`` until ``stop`` is set."""
    while not stop.is_set():
        try:
            await reconcile_unattended(
                repo,
                dispatch,
                cleanup,
                grace_seconds=grace_seconds,
                max_age_seconds=max_age_seconds,
            )
            await purge_finished_work(repo, work_dir)
            if cache_dir:
                await asyncio.to_thread(purge_expired_caches, cache_dir, retention_days)
        except Exception as e:  # noqa: BLE001 - a failed round must not end the loop
            logger.warning(f"[TaxonomyJob] housekeeping round failed: {type(e).__name__}: {e}")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


def _main(argv: List[str]) -> int:
    from airweave.core.config import settings

    if len(argv) != 2 or argv[0] != "delete-tenant":
        print(
            "usage: python -m airweave.domains.taxonomy.jobs.housekeeping delete-tenant <tenant id>"
        )
        return 2
    if not settings.TAXONOMY_JOBS_CACHE_DIR:
        print("TAXONOMY_JOBS_CACHE_DIR is not set: nothing is retained")
        return 0
    removed = delete_tenant_cache(settings.TAXONOMY_JOBS_CACHE_DIR, argv[1])
    print("removed" if removed else "nothing retained for that tenant")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
