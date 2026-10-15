"""Taxonomy domain events (contract 6.2)."""

from typing import Literal, Optional
from uuid import UUID

from airweave.core.events.base import DomainEvent
from airweave.core.events.enums import TaxonomyEventType


class TaxonomyRelabelCompletedEvent(DomainEvent):
    """A re-label job ended. Goes out through the webhook path like ``sync.completed``.

    graph-sync rebuilds the collection's sections when it receives it; the platform does not
    rely on the event alone and checks the chunk statistics itself.
    """

    event_type: TaxonomyEventType = TaxonomyEventType.RELABEL_COMPLETED

    relabel_job_id: UUID
    collection_id: UUID
    collection_readable_id: str
    taxonomy_version: str
    previous_taxonomy_version: Optional[str] = None
    chunks_relabelled: int = 0
    chunks_failed: int = 0
    status: Literal["completed", "failed"]
    error: Optional[str] = None
