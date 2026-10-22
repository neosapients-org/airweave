"""Topic boost for search (contract airweave-neo/1, section 5.1).

A boost, not a filter (decision D14): chunks labelled with the chosen topics move up, and
chunks without labels still come back with their hybrid score.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# The hybrid score is two normalized parts (keyword and semantic, each 0 to 1), so 0 to 2, and a
# full topic match adds the weight times 1. At 0.5 a topic match is worth at most a quarter of the
# hybrid range: it reorders close results and cannot outrank a much better match.
MAX_TOPIC_BOOST_WEIGHT = 0.5


# A registry id such as ``acme:credit.loan_quality``: it is placed in a tensor address.
TopicId = Annotated[
    str, StringConstraints(max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$")
]


class TopicBoost(BaseModel):
    """Rank chunks carrying these registry topic ids higher (Vespa profile hybrid_topic)."""

    topic_ids: list[TopicId] = Field(
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
            f"{MAX_TOPIC_BOOST_WEIGHT}; hybrid scores are 0 to 2, so the boost reorders close "
            "results and cannot bury everything else)."
        ),
    )
