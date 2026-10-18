"""Topic boost for search (contract airweave-neo/1, section 5.1).

A boost, not a filter (decision D14): chunks labelled with the chosen topics move up, and
chunks without labels still come back with their hybrid score.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# An additive boost on a hybrid score of about 0 to 1: above this it would decide the order alone.
MAX_TOPIC_BOOST_WEIGHT = 2.0


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
        le=MAX_TOPIC_BOOST_WEIGHT,
        allow_inf_nan=False,
        description=(
            "Weight of the topic match added to the hybrid score (0 to "
            f"{MAX_TOPIC_BOOST_WEIGHT}; hybrid scores are about 0 to 1, so the boost reorders "
            "close results and cannot bury everything else)."
        ),
    )
