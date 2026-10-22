"""Explicit category status changes (e.g. PROBATIONARY -> MATURE) as a new immutable taxonomy version.

Automatic promotion was evaluated in research (category maturity arm F: trusted-member support, dispersion and waves on
probation) and NOT selected because it did not improve precision. This module therefore offers only an explicit,
reviewed operation; it never decides on its own.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from ..taxonomy import artifacts, maturity


def set_status(taxonomy, store, changes: dict[str, str], new_version: str, reason: str) -> object:
    """changes: {category_id: new status}. Writes taxonomy `new_version` into `store`; the input version is unchanged."""
    final = copy.deepcopy(taxonomy.final)
    by = {c["category_id"]: c for c in final["categories"]}
    for cid, st in changes.items():
        if cid not in by:
            raise KeyError(f"unknown category {cid}")
        if st not in maturity.STATUSES:
            raise ValueError(f"unknown status {st!r}")
        by[cid].setdefault("status_history", []).append({"from": by[cid].get("status", maturity.MATURE), "to": st, "version": new_version,
                                                        "reason": reason})
        by[cid]["status"] = st
    final.update({"taxonomy_version": new_version, "created_at": datetime.now(timezone.utc).isoformat(),
                  "derived_from": {"taxonomy_id": taxonomy.taxonomy_id, "taxonomy_version": taxonomy.version,
                                   "final_taxonomy_sha256": artifacts._sha(taxonomy.path / "final_taxonomy.json")},
                  "status_change": {"changes": changes, "reason": reason}})
    final.pop("artifacts", None)
    out = store.path(taxonomy.taxonomy_id, new_version)
    if out.exists():
        raise FileExistsError(f"{out} already exists; taxonomy versions are immutable")
    rep = artifacts.load_representation(taxonomy)
    artifacts.save(out, {k: v for k, v in final.items() if k != "assignment"}, taxonomy.assigner, taxonomy.bm25,
                   taxonomy.category_centroids, taxonomy.category_sparse, rep.prototypes, rep.exemplars)
    return artifacts.load(out)


def promote(taxonomy, store, category_ids: list[str], new_version: str, reason: str):
    """Explicit PROBATIONARY -> MATURE promotion (a reviewed decision; there is no validated automatic rule)."""
    for cid in category_ids:
        if taxonomy._by_id()[cid].get("status") != maturity.PROBATIONARY:
            raise ValueError(f"{cid} is not PROBATIONARY")
    return set_status(taxonomy, store, {c: maturity.MATURE for c in category_ids}, new_version, reason)
