"""Re-label job: the whole state of one taxonomy re-label (P7 design section 5)."""

from datetime import datetime
from typing import Optional
from uuid import UUID as UUIDType

from sqlalchemy import (
    UUID,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from airweave.models._base import OrganizationBase


class TaxonomyRelabelJob(OrganizationBase):
    """One re-label job. The row is checkpoint, counts and lease, so a restart resumes from it."""

    __tablename__ = "taxonomy_relabel_job"
    __table_args__ = (
        # A replayed start with the same key returns the same job
        UniqueConstraint("collection_id", "idempotency_key", name="uq_relabel_job_key"),
        # One pending or running job per collection
        Index(
            "uq_relabel_job_active_collection",
            "collection_id",
            unique=True,
            postgresql_where=text("state IN ('PENDING', 'RUNNING')"),
        ),
        Index("ix_relabel_job_state", "state"),
    )

    collection_id: Mapped[UUIDType] = mapped_column(
        UUID, ForeignKey("collection.id", ondelete="CASCADE"), nullable=False
    )
    collection_readable_id: Mapped[str] = mapped_column(String, nullable=False)
    target_version: Mapped[str] = mapped_column(String, nullable=False)
    previous_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False, default="full")
    scope: Mapped[str] = mapped_column(String, nullable=False, default="differing")
    idempotency_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    state: Mapped[str] = mapped_column(String, nullable=False, default="PENDING")

    spend_cap_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spend_estimate_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    chunks_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_relabelled: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_stamped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_skipped_changed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_unlabellable: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    checkpoint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    lease_owner: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completion_event_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
