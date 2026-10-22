"""Seed-stability helpers: a cluster "recurs" in another UMAP seed if its best-matching cluster there has
Jaccard >= stability.match_jaccard."""

from __future__ import annotations

import numpy as np


def jac(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def best_match(members: np.ndarray, other_labels: np.ndarray) -> tuple[float, float]:
    """Best Jaccard of `members` against any cluster of another labelling, and the overlap coefficient of that match."""
    best_j, best_o = 0.0, 0.0
    for k2 in set(other_labels[members]) - {-1}:
        m2 = np.where(other_labels == k2)[0]
        j = jac(members, m2)
        if j > best_j:
            best_j, best_o = j, len(set(members) & set(m2)) / min(len(members), len(m2))
    return best_j, best_o
