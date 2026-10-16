"""Taxonomy discover and propose jobs (P7 design sections 6 and 12): one row per job."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID as UUIDType

from sqlalchemy import (
    JSON,
    UUID,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from airweave.models._base import OrganizationBase


class TaxonomyJob(OrganizationBase):
    """A discover (A13) or propose (A12) job. The row is its whole state, so a restart resumes."""

    __tablename__ = "taxonomy_job"
    __table_args__ = (
        UniqueConstraint("collection_id", "idempotency_key", name="uq_taxonomy_job_key"),
        # One pending or running taxonomy job per collection
        Index(
            "uq_taxonomy_job_active_collection",
            "collection_id",
            unique=True,
            postgresql_where=text("state IN ('PENDING', 'RUNNING')"),
        ),
        Index("ix_taxonomy_job_state", "state"),
    )

    kind: Mapped[str] = mapped_column(String, nullable=False)  # discover | propose
    collection_id: Mapped[UUIDType] = mapped_column(
        UUID, ForeignKey("collection.id", ondelete="CASCADE"), nullable=False
    )
    collection_readable_id: Mapped[str] = mapped_column(String, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    candidate_version: Mapped[str] = mapped_column(String, nullable=False)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    params: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    state: Mapped[str] = mapped_column(String, nullable=False, default="PENDING")
    phase: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    spend_cap_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spend_used_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    lease_owner: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
