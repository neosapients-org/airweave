"""Category representation used by learned acceptance and wave evolution (all categories, final_taxonomy.json order).

Per category: single centroid, multi-prototypes (k-means sub-centroids, k in 1..4 by silhouette), exemplar vectors
(category members), sparse BM25 profile, statistics, scope, broad family and status. Plus shared reject prototypes.
Stored with the taxonomy artifact:
  category_centroids.npy, category_sparse_profiles.npz          (also used by novelty)
  category_prototypes.npz, category_exemplars.npz              (stacked vectors + per-category offsets)
  reject prototypes = the rows after the positives in dense_prototypes.npy
Category records in final_taxonomy.json carry `status`, `stats` (cat_members, cat_companies, cat_documents,
cat_dominant_share, cat_dispersion, cat_is_child, cat_is_sink) and `is_sink`. Nothing here knows a domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .maturity import MATURE

MULTI_PROTO = {"candidate_k": (2, 3, 4), "min_members_per_k": 15, "min_cluster": 5, "min_silhouette": 0.10}


@dataclass
class CategoryRepresentation:
    ids: list
    centroids: np.ndarray          # (P, d)
    prototypes: list               # P arrays (k_j, d)
    exemplars: list                # P arrays (m_j, d)
    sparse: sp.csr_matrix          # (P, V)
    reject: np.ndarray             # (R, d)
    stats: list                    # P dicts (STAT_KEYS)
    scopes: list
    families: list
    statuses: list

    def index(self, category_id: str) -> int:
        return self.ids.index(category_id)


def choose_k(Xm: np.ndarray) -> int:
    """Number of sub-prototypes (validated research rule): best cosine silhouette over k=2..4, else 1."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    c = MULTI_PROTO
    best_k, best_s = 1, -1.0
    for k in c["candidate_k"]:
        if len(Xm) < c["min_members_per_k"] * k:
            break
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xm)
        if np.bincount(lab).min() < c["min_cluster"]:
            continue
        s = silhouette_score(Xm, lab, metric="cosine")
        if s > best_s:
            best_k, best_s = k, s
    return best_k if best_s >= c["min_silhouette"] else 1


def subprotos(Xm: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    if k == 1 or len(Xm) < 2 * k:
        p = Xm.mean(0); return (p / np.linalg.norm(p))[None].astype(np.float32)
    km = KMeans(k, n_init=10, random_state=0).fit(Xm)
    P = np.stack([Xm[km.labels_ == j].mean(0) for j in range(k)])
    return (P / np.linalg.norm(P, axis=1, keepdims=True)).astype(np.float32)


def category_stats(Xm: np.ndarray, centroid: np.ndarray, groups, documents, is_child: bool, is_sink: bool) -> dict:
    g = np.asarray(groups, dtype=object)
    vals, counts = np.unique(g, return_counts=True)
    return {"cat_members": int(len(Xm)), "cat_companies": int(len(vals)), "cat_documents": int(len(set(documents))),
            "cat_dominant_share": float(counts.max() / len(Xm)), "cat_dispersion": float(1 - (Xm @ centroid).mean()),
            "cat_is_child": bool(is_child), "cat_is_sink": bool(is_sink)}


def stack(arrays: list, dim: int) -> tuple[np.ndarray, np.ndarray]:
    off = np.cumsum([0] + [len(a) for a in arrays]).astype(np.int64)
    V = np.concatenate([np.asarray(a, np.float32).reshape(-1, dim) for a in arrays]) if arrays else np.zeros((0, dim), np.float32)
    return V, off


def unstack(V: np.ndarray, off: np.ndarray) -> list:
    return [V[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def from_taxonomy(final: dict, centroids, sparse_profiles, prototypes, exemplars, reject) -> CategoryRepresentation:
    cats = final["categories"]
    return CategoryRepresentation(
        ids=[c["category_id"] for c in cats], centroids=np.asarray(centroids, np.float32), prototypes=prototypes, exemplars=exemplars,
        sparse=sp.csr_matrix(sparse_profiles), reject=np.asarray(reject, np.float32),
        stats=[{k: c["stats"][k] for k in ("cat_members", "cat_companies", "cat_documents", "cat_dominant_share", "cat_dispersion",
                                           "cat_is_child", "cat_is_sink")} for c in cats],
        scopes=[c.get("scope") for c in cats], families=[c.get("broad_family_id") for c in cats],
        statuses=[c.get("status", MATURE) for c in cats])
