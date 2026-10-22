"""Multi-scale, multi-seed clustering grid: UMAP (one projection per seed) x HDBSCAN (one run per setting and seed).

Together with candidate_clusters.py this is the ONLY place where cluster membership is decided.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def umap_embed(X: np.ndarray, seed: int, **params) -> np.ndarray:
    """UMAP of the rows of ``X``, the same on every call for the same rows and seed.

    A group of exactly equal rows (the same footer or notice in many documents) at least as large as
    ``n_neighbors`` becomes a separate piece of UMAP's neighbour graph in which every edge has the same
    weight. Its spectral initialisation then has repeated eigenvalues, and the eigensolver returns a
    different, equally valid answer on every call even with a fixed ``random_state``. Each repeat in
    such a group (not its first row) is therefore moved by a tiny fixed amount drawn from ``seed``: the
    tie is broken, and every row stays its own point, so the group keeps its weight and can still form
    a cluster of its own. Rows outside such groups are passed to UMAP unchanged.
    """
    import umap
    _, first, inverse, counts = np.unique(X, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    repeat = np.ones(len(X), bool)
    repeat[first] = False
    repeat &= counts[inverse] >= params.get("n_neighbors", 15)
    if repeat.any():
        X = X.copy()
        noise = np.random.default_rng(seed).standard_normal((int(repeat.sum()), X.shape[1]))
        X[repeat] += (1e-4 * noise).astype(X.dtype)
    return umap.UMAP(random_state=seed, **params).fit_transform(X)


def run_grid(X: np.ndarray, cfg: dict, log=print):
    import hdbscan
    c = cfg["clustering"]
    emb, runs, rows = {}, {}, []
    for s in c["seeds"]:
        t = time.perf_counter()
        emb[s] = umap_embed(X, s, **c["umap"])
        log(f"umap seed {s}: {time.perf_counter() - t:.1f}s")
    for mcs, ms in c["hdbscan_settings"]:
        for s in c["seeds"]:
            t = time.perf_counter()
            lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=c["cluster_selection_method"]).fit_predict(emb[s])
            runs[(mcs, ms, s)] = lab
            rows.append({"min_cluster_size": mcs, "min_samples": ms, "seed": s, "clusters": int(lab.max() + 1),
                         "noise_share": round(float((lab == -1).mean()), 4), "seconds": round(time.perf_counter() - t, 2)})
    return emb, runs, pd.DataFrame(rows)


def top_terms(cv, TF, idx_sets: dict, n: int = 15) -> dict:
    """c-TF-IDF style distinctive terms per group of chunk indices."""
    bg = np.asarray(TF.sum(0)).ravel() + 1.0
    vocab = np.array(cv.get_feature_names_out())
    out = {}
    for k, idx in idx_sets.items():
        g = np.asarray(TF[idx].sum(0)).ravel()
        score = (g / max(g.sum(), 1)) * np.log(bg.sum() / bg)
        out[k] = list(vocab[np.argsort(-score)[:n]])
    return out
