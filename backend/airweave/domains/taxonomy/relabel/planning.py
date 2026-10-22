"""Which chunks a job classifies, which it only stamps, and what it will cost.

``classify``: label from the chunk's own text. ``stamp``: incremental carry-over, the labels stay
and only ``taxonomy_version`` becomes the target. ``skip``: nothing to do.
"""

from __future__ import annotations

import math
from typing import Optional

from airweave.domains.taxonomy.mapping import NOT_LABELLED
from airweave.domains.taxonomy.relabel.chunks import (
    CARRIED_OVER_STATUSES,
    INCREMENTAL_STATUSES,
    ChunkRow,
)
from airweave.domains.taxonomy.relabel.jobs import (
    MODE_INCREMENTAL,
    SCOPE_NOT_LABELLED,
)
from airweave.domains.taxonomy.relabel.stats import ChunkStats

CLASSIFY, STAMP, SKIP, UNLABELLABLE = "classify", "stamp", "skip", "unlabellable"


def decide(row: ChunkRow, target: str, mode: str, scope: str) -> str:
    """What the job does with one chunk."""
    if not row.labellable:
        return UNLABELLABLE
    if scope == SCOPE_NOT_LABELLED:
        return CLASSIFY if row.status == NOT_LABELLED else SKIP
    if row.version == target:
        return SKIP  # a sync already labelled it with the target
    if mode != MODE_INCREMENTAL:
        return CLASSIFY
    if row.status in INCREMENTAL_STATUSES:
        return CLASSIFY
    if row.status in CARRIED_OVER_STATUSES and row.version:
        return STAMP
    return CLASSIFY  # an unknown status or a version-less label cannot be carried over


def chunks_to_classify(stats: ChunkStats, mode: str, scope: str) -> int:
    """How many chunks the job will send to the classifier (the cost driver)."""
    if scope == SCOPE_NOT_LABELLED:
        return stats.not_labelled_chunks
    if mode == MODE_INCREMENTAL:
        return stats.incremental_backlog
    return stats.other_version


def estimate_usd(chunks: int, usd_per_thousand: float) -> float:
    """Embedding cost of classifying ``chunks`` chunks."""
    return round(chunks / 1000 * usd_per_thousand, 6)


def effective_cap(requested: Optional[float], configured: float) -> float:
    """The cap in force: the request can lower the deployment's ceiling, never raise it.

    A ceiling or a request that is not a finite amount from 0 up approves nothing (0): ``NaN``
    compares false with everything, so a ``min`` or a ``spent >= cap`` check would never trip.
    """
    ceiling = configured if math.isfinite(configured) and configured >= 0 else 0.0
    if requested is None:
        return ceiling
    if not math.isfinite(requested) or requested < 0:
        return 0.0
    return min(requested, ceiling)
