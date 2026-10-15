"""The job table through SQLAlchemy.

Every state change is one UPDATE whose WHERE clause carries the rule (lease free or mine, job
still active), so two workers can never both hold a job. The statements are built by plain
functions so their SQL can be checked without a database.

**Not run against a real Postgres in this batch** (no Docker): the tests check the statements'
SQL and the mapping, and the same rules are tested on ``InMemoryJobRepository``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, AsyncContextManager, Callable, List, Optional
from uuid import UUID

from sqlalchemy import Select, Update, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from airweave.db.session import get_db_context
from airweave.domains.taxonomy.relabel.jobs import (
    ACTIVE_STATES,
    CANCELLED,
    COMPLETED,
    FAILED,
    RUNNING,
    ActiveJobExists,
    RelabelJob,
    utcnow,
)
from airweave.models.taxonomy_relabel_job import TaxonomyRelabelJob as Row

_COLUMNS = [c.name for c in Row.__table__.columns if c.name not in ("modified_at",)]


def to_job(row: Row) -> RelabelJob:
    """Row to the job dataclass."""
    values = {name: getattr(row, name) for name in _COLUMNS}
    return RelabelJob(**values)


def to_row(job: RelabelJob) -> Row:
    """Job dataclass to a new row."""
    return Row(**{name: getattr(job, name) for name in _COLUMNS})


def claim_statement(job_id: UUID, owner: str, lease_seconds: int, now: datetime) -> Update:
    """Take the lease: the job is active and the lease is free, mine or expired."""
    return (
        update(Row)
        .where(
            Row.id == job_id,
            Row.state.in_(ACTIVE_STATES),
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
            started_at=func_coalesce(Row.started_at, now),
        )
        .returning(Row)
    )


def func_coalesce(column: Any, value: Any) -> Any:
    """``COALESCE(column, value)``."""
    from sqlalchemy import func

    return func.coalesce(column, value)


def progress_statement(
    job_id: UUID, owner: str, lease_seconds: int, now: datetime, changes: dict
) -> Update:
    """Write progress and renew the lease, only for the lease holder."""
    return (
        update(Row)
        .where(Row.id == job_id, Row.lease_owner == owner, Row.state.in_(ACTIVE_STATES))
        .values(lease_expires_at=now + timedelta(seconds=lease_seconds), **changes)
        .returning(Row)
    )


def finish_statement(
    job_id: UUID, owner: str, state: str, error: Optional[str], now: datetime, changes: dict
) -> Update:
    """End the job and free the lease, only for the lease holder."""
    return (
        update(Row)
        .where(Row.id == job_id, Row.lease_owner == owner, Row.state.in_(ACTIVE_STATES))
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


def cancel_statements(job_id: UUID, now: datetime) -> tuple[Update, Update]:
    """Flag a running job; cancel at once a job whose lease is free or expired."""
    flag = (
        update(Row)
        .where(Row.id == job_id, Row.state.in_(ACTIVE_STATES))
        .values(cancel_requested=True)
    )
    stop = (
        update(Row)
        .where(
            Row.id == job_id,
            Row.state.in_(ACTIVE_STATES),
            or_(Row.lease_expires_at.is_(None), Row.lease_expires_at <= now),
        )
        .values(state=CANCELLED, finished_at=now, lease_owner=None, lease_expires_at=None)
    )
    return flag, stop


def resumable_query(now: datetime) -> Select:
    """Active jobs whose lease is free or expired, oldest first."""
    return (
        select(Row)
        .where(
            Row.state.in_(ACTIVE_STATES),
            or_(Row.lease_expires_at.is_(None), Row.lease_expires_at <= now),
        )
        .order_by(Row.created_at)
    )


def unannounced_query() -> Select:
    """Ended jobs (completed or failed) whose completion event is not recorded as sent."""
    return select(Row).where(
        Row.state.in_((COMPLETED, FAILED)), Row.completion_event_sent.is_(False)
    )


class SqlJobRepository:
    """``JobRepository`` over the ``taxonomy_relabel_job`` table."""

    def __init__(
        self, session: Callable[[], AsyncContextManager[AsyncSession]] = get_db_context
    ) -> None:
        """Use a fresh session per operation."""
        self._session = session

    async def create(self, job: RelabelJob) -> RelabelJob:
        """Insert; the partial unique index turns a second active job into ``ActiveJobExists``."""
        async with self._session() as db:
            db.add(to_row(job))
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()
                active = await self.find_active(job.collection_id)
                if active is not None:
                    raise ActiveJobExists(active) from None
                raise
        return job

    async def get(
        self, job_id: UUID, organization_id: Optional[UUID] = None
    ) -> Optional[RelabelJob]:
        """By id, optionally scoped to the organization."""
        query = select(Row).where(Row.id == job_id)
        if organization_id is not None:
            query = query.where(Row.organization_id == organization_id)
        return await self._one(query)

    async def find_by_key(self, collection_id: UUID, key: str) -> Optional[RelabelJob]:
        """The collection's job started with this key."""
        return await self._one(
            select(Row).where(Row.collection_id == collection_id, Row.idempotency_key == key)
        )

    async def find_active(self, collection_id: UUID) -> Optional[RelabelJob]:
        """The collection's pending or running job."""
        return await self._one(
            select(Row).where(Row.collection_id == collection_id, Row.state.in_(ACTIVE_STATES))
        )

    async def claim(self, job_id: UUID, owner: str, lease_seconds: int) -> Optional[RelabelJob]:
        """See ``claim_statement``."""
        return await self._returning(claim_statement(job_id, owner, lease_seconds, utcnow()))

    async def save_progress(
        self, job_id: UUID, owner: str, lease_seconds: int, **changes: Any
    ) -> Optional[RelabelJob]:
        """See ``progress_statement``."""
        return await self._returning(
            progress_statement(job_id, owner, lease_seconds, utcnow(), changes)
        )

    async def finish(
        self, job_id: UUID, owner: str, state: str, error: Optional[str], **changes: Any
    ) -> Optional[RelabelJob]:
        """See ``finish_statement``."""
        return await self._returning(
            finish_statement(job_id, owner, state, error, utcnow(), changes)
        )

    async def request_cancel(self, job_id: UUID) -> Optional[RelabelJob]:
        """See ``cancel_statements``."""
        flag, stop = cancel_statements(job_id, utcnow())
        async with self._session() as db:
            await db.execute(flag)
            await db.execute(stop)
            await db.commit()
        return await self.get(job_id)

    async def resumable(self) -> List[RelabelJob]:
        """See ``resumable_query``."""
        return await self._many(resumable_query(utcnow()))

    async def unannounced(self) -> List[RelabelJob]:
        """See ``unannounced_query``."""
        return await self._many(unannounced_query())

    async def mark_event_sent(self, job_id: UUID) -> None:
        """Record that the completion event went out."""
        async with self._session() as db:
            await db.execute(update(Row).where(Row.id == job_id).values(completion_event_sent=True))
            await db.commit()

    async def _one(self, query: Select) -> Optional[RelabelJob]:
        async with self._session() as db:
            row = (await db.execute(query)).scalar_one_or_none()
            return to_job(row) if row is not None else None

    async def _many(self, query: Select) -> List[RelabelJob]:
        async with self._session() as db:
            return [to_job(r) for r in (await db.execute(query)).scalars().all()]

    async def _returning(self, statement: Update) -> Optional[RelabelJob]:
        async with self._session() as db:
            row = (await db.execute(statement)).scalar_one_or_none()
            await db.commit()
            return to_job(row) if row is not None else None
