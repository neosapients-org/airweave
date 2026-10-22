"""Lineage between taxonomy children and their parent Airweave chunk, and the chunk-level status rule."""

from __future__ import annotations

from collections import defaultdict

from ..chunking.taxonomy_chunker import ID_SEP
from ..schemas import (ADMINISTRATIVE, ASSIGNED, CHILD_UNRESOLVED, NON_SUBSTANTIVE_CHILD, PARTIALLY_ASSIGNED,
                       UNRESOLVED)


def parent_of(taxonomy_chunk_id: str) -> str:
    """'<airweave chunk_id>::tc007' -> '<airweave chunk_id>'."""
    return taxonomy_chunk_id.rsplit(ID_SEP, 1)[0]


def group_by_parent(children: list[dict]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for c in children:
        by[c["chunk_id"]].append(c)
    return by


def chunk_status(child_statuses: list[str], has_topics: bool) -> str:
    """ASSIGNED / PARTIALLY_ASSIGNED / UNRESOLVED / ADMINISTRATIVE for one Airweave chunk.

    Substantive children are those not bypassed as administrative and not closest to a non-semantic reject prototype.
    """
    substantive = [s for s in child_statuses if s not in NON_SUBSTANTIVE_CHILD]
    if has_topics:
        return PARTIALLY_ASSIGNED if any(s == CHILD_UNRESOLVED for s in substantive) else ASSIGNED
    return UNRESOLVED if substantive else ADMINISTRATIVE
