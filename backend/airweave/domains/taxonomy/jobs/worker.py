"""The taxonomy jobs worker: ``python -m airweave.domains.taxonomy.jobs.worker <build config>``.

A separate Deployment of the same image (see the Helm template ``taxonomy-jobs-worker.yaml``).
It polls the one jobs queue (``limits.JOBS_TASK_QUEUE``) only and registers only the job
workflow and activity. It needs the full classifier install (UMAP, HDBSCAN, numba) and a role
that may write ``taxonomy/tenants/*/candidates/*`` and read the rest of the taxonomy prefix.
``<build config>`` is either the name of a profile the classifier package ships (the chart's
``taxonomyJobsWorker.buildProfile``, e.g. ``small-corpus``) or the path of a YAML file (the chart
mounts ``taxonomyJobsWorker.buildConfig`` and names it). Without one the worker still starts and
runs propose jobs (they use the serving version's own YAML); a discover job is refused before it
reads anything.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import signal
from pathlib import Path
from typing import Any, Optional

from airweave.core.config import settings
from airweave.core.logging import logger
from airweave.domains.taxonomy import limits

# A shipped profile is named, a YAML file is a path: profile names never contain a dot or slash.
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


def load_build_config(build_config: Optional[str]) -> Optional[dict]:
    """The classifier's build settings: a profile the package ships, by name, or a YAML file."""
    import yaml  # type: ignore[import-untyped]

    if not build_config:
        return None
    if _PROFILE_NAME.fullmatch(build_config):
        from taxonomy_classifier import profiles

        logger.info(f"Taxonomy jobs worker builds with the package profile {build_config!r}")
        return yaml.safe_load(profiles.path(build_config).read_text())
    return yaml.safe_load(Path(build_config).read_text())


def build_runner(container: Any, build_config_ref: Optional[str]) -> Any:
    """Assemble the runner from the container's resolver, the settings and the build settings."""
    from airweave.adapters.storage.aws_s3 import S3Backend
    from airweave.domains.taxonomy.artifacts import TAXONOMY_S3_PREFIX
    from airweave.domains.taxonomy.jobs.candidates import CandidateWriter
    from airweave.domains.taxonomy.jobs.gateway import PackageGateway
    from airweave.domains.taxonomy.jobs.records import SqlTaxonomyJobRepository
    from airweave.domains.taxonomy.jobs.runner import TaxonomyJobRunner
    from airweave.domains.taxonomy.relabel.chunk_store import VespaChunkStore

    if not settings.TAXONOMY_S3_BUCKET:
        raise SystemExit("TAXONOMY_S3_BUCKET is required for the taxonomy jobs worker")
    build_config = load_build_config(build_config_ref)
    if build_config is None:
        logger.warning("Taxonomy jobs worker has no classifier build config: discover jobs fail")
    storage = S3Backend(
        bucket=settings.TAXONOMY_S3_BUCKET,
        region=settings.STORAGE_AWS_REGION or "us-east-1",
        endpoint_url=settings.STORAGE_AWS_ENDPOINT_URL,
    )
    return TaxonomyJobRunner(
        SqlTaxonomyJobRepository(),
        VespaChunkStore,
        CandidateWriter(storage, TAXONOMY_S3_PREFIX),
        PackageGateway(build_config, cache_dir=settings.TAXONOMY_JOBS_CACHE_DIR or None),
        container.taxonomy_processor.resolver,
        work_dir=limits.jobs_work_dir(settings.TAXONOMY_DATA_DIR),
        lease_seconds=limits.JOBS_LEASE_SECONDS,
        max_chunks=limits.JOBS_MAX_CHUNKS,
        default_sample_documents=limits.DISCOVER_SAMPLE_DOCUMENTS,
        min_documents=limits.DISCOVER_MIN_DOCUMENTS,
        min_chunks=settings.DISCOVER_MIN_CHUNKS,
    )


def registrations(runner: Any) -> tuple[list, list]:
    """The workflows and activities this worker registers (and nothing else)."""
    from airweave.domains.taxonomy.jobs.temporal import TaxonomyJobActivity, TaxonomyJobWorkflow

    return [TaxonomyJobWorkflow], [TaxonomyJobActivity(runner).run]


async def main(build_config_ref: Optional[str], stop: Optional[asyncio.Event] = None) -> None:
    """Run until stopped; ``build_config_ref`` is a profile name, a YAML path, or None."""
    from temporalio.worker import Worker

    from airweave.core import container as container_mod
    from airweave.core.container import initialize_container
    from airweave.domains.taxonomy.package import (
        JOBS_INTERFACE,
        SYNC_INTERFACE,
        require_classifier_package,
    )
    from airweave.domains.temporal.client import get_client

    if not settings.TAXONOMY_JOBS_ENABLED or not settings.TAXONOMY_ENABLED:
        raise SystemExit("TAXONOMY_ENABLED and TAXONOMY_JOBS_ENABLED must both be true")
    # The jobs need the package's build extra too: refused here, not at the first job.
    require_classifier_package(SYNC_INTERFACE, JOBS_INTERFACE)
    initialize_container(settings)
    runner = build_runner(container_mod.container, build_config_ref)
    workflows, activities = registrations(runner)
    client = await get_client()
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    worker = Worker(
        client,
        task_queue=limits.JOBS_TASK_QUEUE,
        workflows=workflows,
        activities=activities,
        max_concurrent_activities=1,  # one build at a time: it needs the whole pod
    )
    logger.info(f"Taxonomy jobs worker on queue {limits.JOBS_TASK_QUEUE}")
    async with worker:
        housekeeping = asyncio.create_task(_housekeeping(client, stop))
        try:
            await stop.wait()
        finally:
            housekeeping.cancel()
            await asyncio.gather(housekeeping, return_exceptions=True)


async def _housekeeping(client: Any, stop: asyncio.Event) -> None:
    """Start workflows for jobs nothing runs, fail ones abandoned too long, expire old caches."""
    import shutil

    from airweave.domains.taxonomy.jobs.housekeeping import housekeeping_loop
    from airweave.domains.taxonomy.jobs.records import SqlTaxonomyJobRepository
    from airweave.domains.taxonomy.jobs.temporal import start_job_workflow

    async def dispatch(job: Any) -> None:
        await start_job_workflow(client, limits.JOBS_TASK_QUEUE, job.id)

    work_dir = limits.jobs_work_dir(settings.TAXONOMY_DATA_DIR)

    def cleanup(job_id: Any) -> None:
        # Raises on failure (the reconciler logs it); the folder is retried by purge_finished_work.
        folder = Path(work_dir) / str(job_id)
        if folder.exists():
            shutil.rmtree(folder)

    await housekeeping_loop(
        stop,
        SqlTaxonomyJobRepository(),
        dispatch,
        cleanup,
        interval_seconds=limits.JOBS_RECONCILE_SECONDS,
        grace_seconds=limits.JOBS_DISPATCH_GRACE_SECONDS,
        max_age_seconds=limits.JOBS_MAX_AGE_HOURS * 3600,
        cache_dir=settings.TAXONOMY_JOBS_CACHE_DIR or None,
        retention_days=limits.JOBS_CACHE_RETENTION_DAYS,
        work_dir=work_dir,
    )


def _build_config_argument(argv: Optional[list[str]] = None) -> Optional[str]:
    """The build settings named on the command line: a package profile or a YAML path."""
    parser = argparse.ArgumentParser(description="The taxonomy discover and propose jobs worker")
    parser.add_argument(
        "build_config",
        nargs="?",
        default=None,
        help="a classifier package profile (e.g. small-corpus) or the path of its YAML "
        "(a discover job needs one; propose does not)",
    )
    return parser.parse_args(argv).build_config


if __name__ == "__main__":
    asyncio.run(main(_build_config_argument()))
