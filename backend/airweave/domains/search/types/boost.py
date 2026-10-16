"""Topic boost for search (contract airweave-neo/1, section 5.1).

A boost, not a filter (decision D14): chunks labelled with the chosen topics move up, and
chunks without labels still come back with their hybrid score.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicBoost(BaseModel):
    """Rank chunks carrying these registry topic ids higher (Vespa profile hybrid_topic)."""

    topic_ids: list[str] = Field(
        default_factory=list,
        max_length=200,
        description="Registry category ids to boost; chunks with any of them rank higher.",
    )
    weight: float = Field(
        default=0.15,
        ge=0.0,
        description="Weight of the topic match added to the hybrid score.",
    )
