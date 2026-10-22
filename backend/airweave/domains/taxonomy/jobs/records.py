"""The discover and propose job record and its repository (A13, A12).

The row is the whole state of a job. A worker holds it under a lease that it renews while it
works (every heartbeat); a lease that expires frees the job for the next attempt, which is how a
job resumes after a crash. Every write a running job makes is fenced by its lease.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol
from uuid import UUID

from sqlalchemy import Update, or_, select, update
from sqlalchemy.exc import IntegrityError

from airweave.models.taxonomy_job import TaxonomyJob as Row

PENDING, RUNNING, COMPLETED, FAILED, CANCELLED = (
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
)
ACTIVE = (PENDING, RUNNING)
KIND_DISCOVER, KIND_PROPOSE = "discover", "propose"


def utcnow() -> datetime:
    """Naive UTC now, like the other tables."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class TaxonomyJobRecord:
    """A discover or propose job."""

    id: UUID
    organization_id: UUID
    kind: str
    collection_id: UUID
    collection_readable_id: str
    tenant_id: str
    candidate_version: str
    idempotency_key: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    state: str = PENDING
    phase: Optional[str] = None
    spend_cap_usd: float = 0.0
    spend_used_usd: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def active(self) -> bool:
        """Pending or running."""
        return self.state in ACTIVE


class ActiveTaxonomyJob(Exception):
    """The collection already has a pending or running taxonomy job."""

    def __init__(self, job: TaxonomyJobRecord) -> None:
        """Keep the blocking job."""
        super().__init__(f"collection {job.collection_readable_id} has an active job {job.id}")
        self.job = job


class TaxonomyJobRepository(Protocol):
    """Persistence of taxonomy jobs."""

    async def create(self, job: TaxonomyJobRecord) -> TaxonomyJobRecord:
        """Insert; raises ``ActiveTaxonomyJob`` if the collection has an active job."""
        ...

    async def get(
        self, job_id: UUID, organization_id: Optional[UUID] = None
    ) -> Optional[TaxonomyJobRecord]:
        """By id, optionally only inside the organization."""
        ...

    async def find_by_key(self, collection_id: UUID, key: str) -> Optional[TaxonomyJobRecord]:
        """The collection's job started with this idempotency key."""
        ...

    async def find_active(self, collection_id: UUID) -> Optional[TaxonomyJobRecord]:
        """The collection's pending or running job."""
        ...

    async def claim(
        self, job_id: UUID, owner: str, lease_seconds: int
    ) -> Optional[TaxonomyJobRecord]:
        """Take the lease if the job is active and the lease is free, mine or expired."""
        ...

    async def renew(
        self, job_id: UUID, owner: str, lease_seconds: int, **changes: Any
    ) -> Optional[TaxonomyJobRecord]:
        """Renew the lease and save progress; ``None`` if the lease was lost."""
        ...

    async def finish(
        self, job_id: UUID, owner: str, state: str, error: Optional[str], **changes: Any
    ) -> Optional[TaxonomyJobRecord]:
        """End the job under the lease; ``None`` if the lease was lost."""
        ...

    async def request_cancel(self, job_id: UUID) -> Optional[TaxonomyJobRecord]:
        """Flag a running job; cancel at once a job nobody holds."""
        ...

    async def list_unattended(
        self, older_than: datetime, limit: int = 50
    ) -> List[TaxonomyJobRecord]:
        """Active jobs created before ``older_than`` that no live lease holds, oldest first.

        These are the jobs a worker died on, whose workflow ended, or whose dispatch never
        reached Temporal; the worker's reconciler starts a workflow for each.
        """
        ...

    async def abandon(
        self, job_id: UUID, error: str, older_than: datetime
    ) -> Optional[TaxonomyJobRecord]:
        """Fail an active, unattended job created before ``older_than``; ``None`` if it is not."""
        ...


class InMemoryTaxonomyJobRepository:
    """The repository's rules in memory (tests)."""

    def __init__(self, clock: Any = utcnow) -> None:
        """Start empty."""
        self.jobs: Dict[UUID, TaxonomyJobRecord] = {}
        self._clock = clock

    async def create(self, job: TaxonomyJobRecord) -> TaxonomyJobRecord:
        """Insert a copy."""
        active = await self.find_active(job.collection_id)
        if active is not None:
            raise ActiveTaxonomyJob(active)
        self.jobs[job.id] = replace(job)
        return replace(job)

    async def get(
        self, job_id: UUID, organization_id: Optional[UUID] = None
    ) -> Optional[TaxonomyJobRecord]:
        """A copy."""
        job = self.jobs.get(job_id)
        if job is None or (organization_id and job.organization_id != organization_id):
            return None
        return replace(job)

    async def find_by_key(self, collection_id: UUID, key: str) -> Optional[TaxonomyJobRecord]:
        """By collection and key."""
        for job in self.jobs.values():
            if job.collection_id == collection_id and job.idempotency_key == key:
                return replace(job)
        return None

    async def find_active(self, collection_id: UUID) -> Optional[TaxonomyJobRecord]:
        """The collection's active job."""
        for job in self.jobs.values():
            if job.collection_id == collection_id and job.active:
                return replace(job)
        return None

    def _free(self, job: TaxonomyJobRecord, owner: str) -> bool:
        return (
            job.lease_owner in (None, owner)
            or job.lease_expires_at is None
            or job.lease_expires_at <= self._clock()
        )

    async def claim(
        self, job_id: UUID, owner: str, lease_seconds: int
    ) -> Optional[TaxonomyJobRecord]:
        """Atomic in memory."""
        job = self.jobs.get(job_id)
        if job is None or not job.active or not self._free(job, owner):
            return None
        job.state, job.lease_owner = RUNNING, owner
        job.lease_expires_at = self._clock() + timedelta(seconds=lease_seconds)
        job.started_at = job.started_at or self._clock()
        return replace(job)

    async def renew(
        self, job_id: UUID, owner: str, lease_seconds: int, **changes: Any
    ) -> Optional[TaxonomyJobRecord]:
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
    ) -> Optional[TaxonomyJobRecord]:
        """Fenced by the lease owner."""
        job = self.jobs.get(job_id)
        if job is None or job.lease_owner != owner or not job.active:
            return None
        for name, value in changes.items():
            setattr(job, name, value)
        job.state, job.error, job.finished_at = state, error, self._clock()
        job.lease_owner = job.lease_expires_at = None
        return replace(job)

    def _unattended(self, job: TaxonomyJobRecord) -> bool:
        return job.active and (
            job.lease_expires_at is None or job.lease_expires_at <= self._clock()
        )

    async def list_unattended(
        self, older_than: datetime, limit: int = 50
    ) -> List[TaxonomyJobRecord]:
        """Oldest first."""
        found = [
            replace(j)
            for j in self.jobs.values()
            if self._unattended(j) and j.created_at <= older_than
        ]
        return sorted(found, key=lambda j: j.created_at)[:limit]

    async def abandon(
        self, job_id: UUID, error: str, older_than: datetime
    ) -> Optional[TaxonomyJobRecord]:
        """Only an unattended job created before ``older_than``."""
        job = self.jobs.get(job_id)
        if job is None or not self._unattended(job) or job.created_at > older_than:
            return None
        job.state, job.error, job.finished_at = FAILED, error, self._clock()
        job.lease_owner = job.lease_expires_at = None
        return replace(job)

    async def request_cancel(self, job_id: UUID) -> Optional[TaxonomyJobRecord]:
        """Flag it, or cancel at once without a live lease."""
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.active:
            job.cancel_requested = True
            held = job.lease_owner and job.lease_expires_at and job.lease_expires_at > self._clock()
            if not held:
                job.state, job.finished_at = CANCELLED, self._clock()
                job.lease_owner = job.lease_expires_at = None
        return replace(job)


_COLUMNS = [c.name for c in Row.__table__.columns if c.name != "modified_at"]


def to_record(row: Row) -> TaxonomyJobRecord:
    """Row to record."""
    values = {name: getattr(row, name) for name in _COLUMNS}
    values["params"] = values["params"] or {}
    return TaxonomyJobRecord(**values)


def to_row(job: TaxonomyJobRecord) -> Row:
    """Record to a new row."""
    return Row(**{name: getattr(job, name) for name in _COLUMNS})


def claim_statement(job_id: UUID, owner: str, lease_seconds: int, now: datetime) -> Update:
    """Take the lease: active job, lease free, mine or expired."""
    from sqlalchemy import func

    return (
        update(Row)
        .where(
            Row.id == job_id,
            Row.state.in_(ACTIVE),
            or_(
                Row.lease_owner.is_(None),
                Row.lease_owner == owner,
                Row.lease_expires_at.is_(None),
                Row.lease_expires_at <= now,
            ),
        )
        .values(
            state=RUNNING,
            lease_owner=owner,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            started_at=func.coalesce(Row.started_at, now),
        )
        .returning(Row)
    )


def renew_statement(
    job_id: UUID, owner: str, lease_seconds: int, now: datetime, changes: dict
) -> Update:
    """Renew and save progress, only for the lease holder."""
    return (
        update(Row)
        .where(Row.id == job_id, Row.lease_owner == owner, Row.state.in_(ACTIVE))
        .values(lease_expires_at=now + timedelta(seconds=lease_seconds), **changes)
        .returning(Row)
    )


def finish_statement(
    job_id: UUID, owner: str, state: str, error: Optional[str], now: datetime, changes: dict
) -> Update:
    """End the job and free the lease, only for the lease holder."""
    return (
        update(Row)
        .where(Row.id == job_id, Row.lease_owner == owner, Row.state.in_(ACTIVE))
        .values(
            state=state,
            error=error,
            finished_at=now,
            lease_owner=None,
            lease_expires_at=None,
            **changes,
        )
        .returning(Row)
    )


def unattended_clause(now: datetime) -> Any:
    """Active, and no live lease holds it."""
    from sqlalchemy import and_

    return and_(
        Row.state.in_(ACTIVE),
        or_(Row.lease_expires_at.is_(None), Row.lease_expires_at <= now),
    )


def cancel_statements(job_id: UUID, now: datetime) -> tuple[Update, Update]:
    """Flag a running job; cancel at once one whose lease is free or expired."""
    flag = update(Row).where(Row.id == job_id, Row.state.in_(ACTIVE)).values(cancel_requested=True)
    stop = (
        update(Row)
        .where(
            Row.id == job_id,
            Row.state.in_(ACTIVE),
            or_(Row.lease_expires_at.is_(None), Row.lease_expires_at <= now),
        )
        .values(state=CANCELLED, finished_at=now, lease_owner=None, lease_expires_at=None)
    )
    return flag, stop


class SqlTaxonomyJobRepository:
    """``TaxonomyJobRepository`` over the ``taxonomy_job`` table (not run on a real Postgres)."""

    def __init__(self, session: Any = None) -> None:
        """Use a fresh session per operation."""
        if session is None:
            from airweave.db.session import get_db_context

            session = get_db_context
        self._session = session

    async def create(self, job: TaxonomyJobRecord) -> TaxonomyJobRecord:
        """Insert; the partial unique index makes a second active job an ``ActiveTaxonomyJob``."""
        async with self._session() as db:
            db.add(to_row(job))
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()
                active = await self.find_active(job.collection_id)
                if active is not None:
                    raise ActiveTaxonomyJob(active) from None
                raise
        return job

    async def get(
        self, job_id: UUID, organization_id: Optional[UUID] = None
    ) -> Optional[TaxonomyJobRecord]:
        """By id."""
        query = select(Row).where(Row.id == job_id)
        if organization_id is not None:
            query = query.where(Row.organization_id == organization_id)
        return await self._one(query)

    async def find_by_key(self, collection_id: UUID, key: str) -> Optional[TaxonomyJobRecord]:
        """By collection and key."""
        return await self._one(
            select(Row).where(Row.collection_id == collection_id, Row.idempotency_key == key)
        )

    async def find_active(self, collection_id: UUID) -> Optional[TaxonomyJobRecord]:
        """The collection's active job."""
        return await self._one(
            select(Row).where(Row.collection_id == collection_id, Row.state.in_(ACTIVE))
        )

    async def claim(
        self, job_id: UUID, owner: str, lease_seconds: int
    ) -> Optional[TaxonomyJobRecord]:
        """See ``claim_statement``."""
        return await self._returning(claim_statement(job_id, owner, lease_seconds, utcnow()))

    async def renew(
        self, job_id: UUID, owner: str, lease_seconds: int, **changes: Any
    ) -> Optional[TaxonomyJobRecord]:
        """See ``renew_statement``."""
        return await self._returning(
            renew_statement(job_id, owner, lease_seconds, utcnow(), changes)
        )

    async def finish(
        self, job_id: UUID, owner: str, state: str, error: Optional[str], **changes: Any
    ) -> Optional[TaxonomyJobRecord]:
        """See ``finish_statement``."""
        return await self._returning(
            finish_statement(job_id, owner, state, error, utcnow(), changes)
        )

    async def request_cancel(self, job_id: UUID) -> Optional[TaxonomyJobRecord]:
        """See ``cancel_statements``."""
        flag, stop = cancel_statements(job_id, utcnow())
        async with self._session() as db:
            await db.execute(flag)
            await db.execute(stop)
            await db.commit()
        return await self.get(job_id)

    async def list_unattended(
        self, older_than: datetime, limit: int = 50
    ) -> List[TaxonomyJobRecord]:
        """See ``unattended_clause``."""
        query = (
            select(Row)
            .where(unattended_clause(utcnow()), Row.created_at <= older_than)
            .order_by(Row.created_at)
            .limit(limit)
        )
        async with self._session() as db:
            return [to_record(r) for r in (await db.execute(query)).scalars().all()]

    async def abandon(
        self, job_id: UUID, error: str, older_than: datetime
    ) -> Optional[TaxonomyJobRecord]:
        """Fail an unattended job created before ``older_than``."""
        now = utcnow()
        return await self._returning(
            update(Row)
            .where(Row.id == job_id, unattended_clause(now), Row.created_at <= older_than)
            .values(
                state=FAILED, error=error, finished_at=now, lease_owner=None, lease_expires_at=None
            )
            .returning(Row)
        )

    async def _one(self, query: Any) -> Optional[TaxonomyJobRecord]:
        async with self._session() as db:
            row = (await db.execute(query)).scalar_one_or_none()
            return to_record(row) if row is not None else None

    async def _returning(self, statement: Update) -> Optional[TaxonomyJobRecord]:
        async with self._session() as db:
            row = (await db.execute(statement)).scalar_one_or_none()
            # Read the row before the commit: a commit expires it, and reading an expired
            # attribute outside the greenlet raises MissingGreenlet (found on a real database).
            record = to_record(row) if row is not None else None
            await db.commit()
            return record


__all__: List[str] = [
    "ACTIVE",
    "ActiveTaxonomyJob",
    "CANCELLED",
    "COMPLETED",
    "FAILED",
    "InMemoryTaxonomyJobRepository",
    "KIND_DISCOVER",
    "KIND_PROPOSE",
    "PENDING",
    "RUNNING",
    "SqlTaxonomyJobRepository",
    "TaxonomyJobRecord",
    "TaxonomyJobRepository",
]
