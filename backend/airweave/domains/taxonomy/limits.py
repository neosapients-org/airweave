"""The taxonomy's fixed numbers and folder layout.

These were settings; nobody sets them per deployment, and a wrong value is a bug, not a
configuration, so they are constants. What a deployment does decide is in ``Settings``:
``TAXONOMY_ENABLED``, ``TAXONOMY_S3_BUCKET``, ``TAXONOMY_DATA_DIR``, ``TAXONOMY_CLASSIFY_THREADS``,
``TAXONOMY_SPEND_CAP_USD`` and the jobs' switches.
"""

from __future__ import annotations

from pathlib import Path

# --- labelling during sync ---------------------------------------------------------------------
# How long a collection's active-version answer (a failure included) is cached.
ACTIVE_CACHE_SECONDS = 300
# A classifier thread starts a new embedding cache file past this size.
EMBED_CACHE_MAX_MB = 512
# The most one taxonomy file, and one version's files together, may weigh. A manifest that
# declares more is refused before anything is downloaded; the bytes read are checked too.
ARTIFACT_MAX_MB = 256
ARTIFACT_TOTAL_MAX_MB = 1024
# A classification batch that takes longer is marked ERROR and the sync goes on.
CLASSIFY_TIMEOUT_SECONDS = 60

# --- re-label jobs -----------------------------------------------------------------------------
RELABEL_USD_PER_THOUSAND_CHUNKS = 0.006  # embedding cost of classifying 1,000 chunks
RELABEL_BATCH_SIZE = 64  # chunks per classifier call; each batch is charged before it is sent
RELABEL_PAGE_SIZE = 100  # chunks per Vespa visit page (also the unresolved scroll's)
RELABEL_WRITE_CONCURRENCY = 8  # parallel conditional updates
RELABEL_LEASE_SECONDS = 120  # a running job renews its lease; an expired one is resumed
RELABEL_RESUME_INTERVAL_SECONDS = 60  # how often a process looks for jobs to resume

# --- discover and propose jobs -----------------------------------------------------------------
# The one queue only the jobs worker polls. A constant, so the API that enqueues and the worker
# that runs can never be configured onto different queues.
JOBS_TASK_QUEUE = "airweave-taxonomy-jobs-queue"
JOBS_LEASE_SECONDS = 300
JOBS_MAX_CHUNKS = 20_000  # chunks of text one job reads into memory at most
DISCOVER_SAMPLE_DOCUMENTS = 200  # documents sampled when the request gives no size
DISCOVER_MIN_DOCUMENTS = 1  # fewer substantive documents: the job fails, nothing is spent
# Days a tenant's retained cache folder may sit unused before the jobs worker removes it.
JOBS_CACHE_RETENTION_DAYS = 30
# The jobs worker looks for active jobs nothing runs every RECONCILE seconds, starts a workflow
# for those older than the grace period, and fails those unattended for MAX_AGE hours.
JOBS_RECONCILE_SECONDS = 60
JOBS_DISPATCH_GRACE_SECONDS = 120
JOBS_MAX_AGE_HOURS = 24


def versions_dir(data_dir: str) -> str:
    """Where verified taxonomy versions are kept on disk."""
    return str(Path(data_dir) / "versions")


def embed_cache_dir(data_dir: str) -> str:
    """Where the classifier threads keep their SQLite embedding caches."""
    return str(Path(data_dir) / "embed-cache")


def jobs_work_dir(data_dir: str) -> str:
    """The jobs worker's scratch space; it holds customer text for the length of a job."""
    return str(Path(data_dir) / "jobs")
