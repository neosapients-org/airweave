"""The taxonomy jobs worker: ``python -m airweave.domains.taxonomy.jobs.worker``.

A separate Deployment of the same image (see the Helm template ``taxonomy-jobs-worker.yaml``).
It polls ``TAXONOMY_JOBS_TASK_QUEUE`` only and registers only the job workflow and activity. It
needs the full classifier install (UMAP, HDBSCAN, numba) and a role that may write
``taxonomy/tenants/*/candidates/*`` and read the rest of the taxonomy prefix.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any, Optional

from airweave.core.config import settings
from airweave.core.logging import logger


def build_runner(container: Any) -> Any:
    """Assemble the runner from the container's taxonomy resolver and the settings."""
    import yaml  # type: ignore[import-untyped]

    from airweave.adapters.storage.aws_s3 import S3Backend
    from airweave.domains.taxonomy.jobs.candidates import CandidateWriter
    from airweave.domains.taxonomy.jobs.gateway import PackageGateway
    from airweave.domains.taxonomy.jobs.records import SqlTaxonomyJobRepository
    from airweave.domains.taxonomy.jobs.runner import TaxonomyJobRunner
    from airweave.domains.taxonomy.relabel.chunk_store import VespaChunkStore

    if not settings.TAXONOMY_S3_BUCKET:
        raise SystemExit("TAXONOMY_S3_BUCKET is required for the taxonomy jobs worker")
    if not settings.TAXONOMY_JOBS_BUILD_CONFIG:
        raise SystemExit("TAXONOMY_JOBS_BUILD_CONFIG (the classifier package's YAML) is required")
    build_config = yaml.safe_load(Path(settings.TAXONOMY_JOBS_BUILD_CONFIG).read_text())
    storage = S3Backend(
        bucket=settings.TAXONOMY_S3_BUCKET,
        region=settings.STORAGE_AWS_REGION or "us-east-1",
        endpoint_url=settings.STORAGE_AWS_ENDPOINT_URL,
    )
    return TaxonomyJobRunner(
        SqlTaxonomyJobRepository(),
        VespaChunkStore,
        CandidateWriter(storage, settings.TAXONOMY_S3_PREFIX),
        PackageGateway(build_config),
        container.taxonomy_processor.resolver,
        work_dir=settings.TAXONOMY_JOBS_WORK_DIR,
        lease_seconds=settings.TAXONOMY_JOBS_LEASE_SECONDS,
        max_chunks=settings.TAXONOMY_JOBS_MAX_CHUNKS,
        default_sample_documents=settings.DISCOVER_SAMPLE_DOCUMENTS,
        min_documents=settings.DISCOVER_MIN_DOCUMENTS,
        min_chunks=settings.DISCOVER_MIN_CHUNKS,
    )


def registrations(runner: Any) -> tuple[list, list]:
    """The workflows and activities this worker registers (and nothing else)."""
    from airweave.domains.taxonomy.jobs.temporal import TaxonomyJobActivity, TaxonomyJobWorkflow

    return [TaxonomyJobWorkflow], [TaxonomyJobActivity(runner).run]


async def main(stop: Optional[asyncio.Event] = None) -> None:
    """Run until stopped."""
    from temporalio.worker import Worker

    from airweave.core import container as container_mod
    from airweave.core.container import initialize_container
    from airweave.domains.temporal.client import get_client

    if not settings.TAXONOMY_JOBS_ENABLED or not settings.TAXONOMY_ENABLED:
        raise SystemExit("TAXONOMY_ENABLED and TAXONOMY_JOBS_ENABLED must both be true")
    initialize_container(settings)
    runner = build_runner(container_mod.container)
    workflows, activities = registrations(runner)
    client = await get_client()
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    worker = Worker(
        client,
        task_queue=settings.TAXONOMY_JOBS_TASK_QUEUE,
        workflows=workflows,
        activities=activities,
        max_concurrent_activities=1,  # one build at a time: it needs the whole pod
    )
    logger.info(f"Taxonomy jobs worker on queue {settings.TAXONOMY_JOBS_TASK_QUEUE}")
    async with worker:
        await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
