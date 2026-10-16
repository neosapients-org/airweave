"""The re-label job record and the repository it lives in (P7 design section 5).

One row per job. The row is the whole state of a job, so a worker restart resumes from it: the
scroll checkpoint, the counts, and a lease that a running job renews and an expired one loses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol
from uuid import UUID

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
ACTIVE_STATES = (PENDING, RUNNING)
FINISHED_STATES = (COMPLETED, FAILED, CANCELLED)

MODE_FULL = "full"
MODE_INCREMENTAL = "incremental"
SCOPE_DIFFERING = "differing"
SCOPE_NOT_LABELLED = "not_labelled"

COUNT_FIELDS = (
    "chunks_relabelled",
    "chunks_stamped",
    "chunks_skipped_changed",
    "chunks_unlabellable",
    "chunks_failed",
)


def utcnow() -> datetime:
    """Timezone-aware now (the job table stores naive UTC like the other tables)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class RelabelJob:
    """A re-label job."""

    id: UUID
    organization_id: UUID
    collection_id: UUID
    collection_readable_id: str
    target_version: str
    mode: str = MODE_FULL
    scope: str = SCOPE_DIFFERING
    idempotency_key: Optional[str] = None
    state: str = PENDING
    previous_version: Optional[str] = None
    spend_cap_usd: float = 0.0
    spend_estimate_usd: float = 0.0
    chunks_total: int = 0  # estimate taken when the job was accepted
    chunks_relabelled: int = 0  # classified and written
    chunks_stamped: int = 0  # incremental: carried over, version stamped without classifying
    chunks_skipped_changed: int = 0  # a sync replaced or deleted the chunk: its own labels win
    chunks_unlabellable: int = 0  # code files and chunks without positions
    chunks_failed: int = 0  # could not be classified or written; left unchanged
    checkpoint: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    completion_event_sent: bool = False
    created_at: datetime = field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def active(self) -> bool:
        """Pending or running."""
        return self.state in ACTIVE_STATES

    @property
    def relabelled_in_total(self) -> int:
        """Chunks that now carry the target version through this job (classified or stamped)."""
        return self.chunks_relabelled + self.chunks_stamped

    def counts(self) -> Dict[str, int]:
        """The progress counters."""
        return {name: getattr(self, name) for name in COUNT_FIELDS}


class ActiveJobExists(Exception):
    """The collection already has a pending or running job."""

    def __init__(self, job: RelabelJob) -> None:
        """Keep the job that blocks the new one."""
        super().__init__(f"collection {job.collection_readable_id} has an active job {job.id}")
        self.job = job


class JobRepository(Protocol):
    """Persistence of re-label jobs. Every write that a running job makes is fenced by its lease."""

    async def create(self, job: RelabelJob) -> RelabelJob:
        """Insert; raises ``ActiveJobExists`` if the collection has an active job."""
        ...

    async def get(
        self, job_id: UUID, organization_id: Optional[UUID] = None
    ) -> Optional[RelabelJob]:
        """The job, optionally only if it belongs to the organization."""
        ...

    async def find_by_key(self, collection_id: UUID, key: str) -> Optional[RelabelJob]:
        """The collection's job started with this idempotency key."""
        ...

    async def find_active(self, collection_id: UUID) -> Optional[RelabelJob]:
        """The collection's pending or running job."""
        ...

    async def claim(self, job_id: UUID, owner: str, lease_seconds: int) -> Optional[RelabelJob]:
        """Take the lease if the job is active and its lease is free, mine or expired."""
        ...

    async def save_progress(
        self, job_id: UUID, owner: str, lease_seconds: int, **changes: Any
    ) -> Optional[RelabelJob]:
        """Write counts and checkpoint and renew the lease; ``None`` if the lease was lost."""
        ...

    async def finish(
        self, job_id: UUID, owner: str, state: str, error: Optional[str], **changes: Any
    ) -> Optional[RelabelJob]:
        """End the job under the lease; ``None`` if the lease was lost."""
        ...

    async def request_cancel(self, job_id: UUID) -> Optional[RelabelJob]:
        """Flag a running job for cancellation; cancel a job nobody holds at once."""
        ...

    async def resumable(self) -> List[RelabelJob]:
        """Active jobs whose lease is free or expired, oldest first."""
        ...

    async def unannounced(self) -> List[RelabelJob]:
        """Completed or failed jobs whose completion event was not recorded as sent."""
        ...

    async def mark_event_sent(self, job_id: UUID) -> None:
        """Record that the completion event was published."""
        ...


class InMemoryJobRepository:
    """The repository's behaviour in memory, for tests and for the fake Vespa suites."""

    def __init__(self, clock: Any = utcnow) -> None:
        """Start empty."""
        self.jobs: Dict[UUID, RelabelJob] = {}
        self._clock = clock

    async def create(self, job: RelabelJob) -> RelabelJob:
        """Insert a copy; the same one-active-job-per-collection rule as the table's index."""
        active = await self.find_active(job.collection_id)
        if active is not None:
            raise ActiveJobExists(active)
        if job.idempotency_key and await self.find_by_key(job.collection_id, job.idempotency_key):
            raise ValueError("idempotency key already used for this collection")
        self.jobs[job.id] = replace(job)
        return replace(job)

    async def get(
        self, job_id: UUID, organization_id: Optional[UUID] = None
    ) -> Optional[RelabelJob]:
        """A copy of the job."""
        job = self.jobs.get(job_id)
        if job is None or (organization_id and job.organization_id != organization_id):
            return None
        return replace(job)

    async def find_by_key(self, collection_id: UUID, key: str) -> Optional[RelabelJob]:
        """By collection and key."""
        for job in self.jobs.values():
            if job.collection_id == collection_id and job.idempotency_key == key:
                return replace(job)
        return None

    async def find_active(self, collection_id: UUID) -> Optional[RelabelJob]:
        """The collection's active job."""
        for job in self.jobs.values():
            if job.collection_id == collection_id and job.active:
                return replace(job)
        return None

    def _free(self, job: RelabelJob, owner: str) -> bool:
        return (
            job.lease_owner in (None, owner)
            or job.lease_expires_at is None
            or job.lease_expires_at <= self._clock()
        )

    async def claim(self, job_id: UUID, owner: str, lease_seconds: int) -> Optional[RelabelJob]:
        """Atomic in memory (no awaits between the check and the write)."""
        job = self.jobs.get(job_id)
        if job is None or not job.active or not self._free(job, owner):
            return None
        job.state = RUNNING
        job.lease_owner = owner
        job.lease_expires_at = self._clock() + timedelta(seconds=lease_seconds)
        job.started_at = job.started_at or self._clock()
        return replace(job)

    async def save_progress(
        self, job_id: UUID, owner: str, lease_seconds: int, **changes: Any
    ) -> Optional[RelabelJob]:
        """Fenced by the lease owner."""
        job = self.jobs.get(job_id)
        if job is None or job.lease_owner != owner or not job.active:
            return None
        for name, value in changes.items():
            setattr(job, name, value)
        job.lease_expires_at = self._clock() + timedelta(seconds=lease_seconds)
        return replace(job)

    async def finish(
        self, job_id: UUID, owner: str, state: str, error: Optional[str], **changes: Any
    ) -> Optional[RelabelJob]:
        """Fenced by the lease owner; clears the lease."""
        job = self.jobs.get(job_id)
        if job is None or job.lease_owner != owner or not job.active:
            return None
        for name, value in changes.items():
            setattr(job, name, value)
        job.state, job.error = state, error
        job.finished_at = self._clock()
        job.lease_owner, job.lease_expires_at = None, None
        return replace(job)

    async def request_cancel(self, job_id: UUID) -> Optional[RelabelJob]:
        """Flag it, or cancel at once if no live lease holds it."""
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.active:
            job.cancel_requested = True
            held = job.lease_owner and job.lease_expires_at and job.lease_expires_at > self._clock()
            if not held:
                job.state, job.finished_at = CANCELLED, self._clock()
                job.lease_owner, job.lease_expires_at = None, None
        return replace(job)

    async def resumable(self) -> List[RelabelJob]:
        """Active with a free or expired lease."""
        now = self._clock()
        free = [
            j
            for j in self.jobs.values()
            if j.active and (j.lease_expires_at is None or j.lease_expires_at <= now)
        ]
        return [replace(j) for j in sorted(free, key=lambda j: j.created_at)]

    async def unannounced(self) -> List[RelabelJob]:
        """Finished with an event still to send."""
        return [
            replace(j)
            for j in self.jobs.values()
            if j.state in (COMPLETED, FAILED) and not j.completion_event_sent
        ]

    async def mark_event_sent(self, job_id: UUID) -> None:
        """Flag it."""
        self.jobs[job_id].completion_event_sent = True
