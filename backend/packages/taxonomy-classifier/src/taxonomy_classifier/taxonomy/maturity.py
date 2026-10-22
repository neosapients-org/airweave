"""Category lifecycle status and the probation gate (validated research policy, data-driven by `category.status`).

Statuses: MATURE (normal acceptance), PROBATIONARY (new category: stricter gate), and the retired statuses FROZEN,
REJECTED, MERGED (never assigned). The gate never looks at a category name or a domain.

Validated policy (category_maturity, selected arm D, FINAL_CATEGORY_MATURITY_POLICY.json):
  mature:        P >= model threshold (0.80 for the reference model)
  probationary:  P >= 0.95 and BM25 rank <= 2 and dense (multi-prototype) rank <= 2
  the gate is applied BEFORE the per-child top-1 selection, and also to novelty mappings into a probationary category.
Automatic promotion / freezing (arm F) was evaluated and NOT selected: promotion is an explicit operation only.
"""

from __future__ import annotations

MATURE, PROBATIONARY, FROZEN, REJECTED, MERGED = "MATURE", "PROBATIONARY", "FROZEN", "REJECTED", "MERGED"
STATUSES = (MATURE, PROBATIONARY, FROZEN, REJECTED, MERGED)
RETIRED = (FROZEN, REJECTED, MERGED)
VALIDATED_PROBATION_GATE = {"p_min": 0.95, "bm25_rank_max": 2, "dense_rank_max": 2}


def status_of(category: dict) -> str:
    s = category.get("status", MATURE)
    if s not in STATUSES:
        raise ValueError(f"category {category.get('category_id')}: unknown status {s!r} (allowed {STATUSES})")
    return s


def gate_ok(p: float, bm25_rank: int, dense_rank: int, status: str, threshold: float, probation: dict = VALIDATED_PROBATION_GATE) -> bool:
    if status in RETIRED:
        return False
    if status == MATURE:
        return p >= threshold
    return p >= probation["p_min"] and bm25_rank <= probation["bm25_rank_max"] and dense_rank <= probation["dense_rank_max"]
