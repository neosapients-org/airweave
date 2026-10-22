"""Multi-scale, multi-seed clustering grid: UMAP (one projection per seed) x HDBSCAN (one run per setting and seed).

Together with candidate_clusters.py this is the ONLY place where cluster membership is decided.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def run_grid(X: np.ndarray, cfg: dict, log=print):
    import hdbscan
    import umap
    c = cfg["clustering"]
    emb, runs, rows = {}, {}, []
    for s in c["seeds"]:
        t = time.perf_counter()
        emb[s] = umap.UMAP(random_state=s, **c["umap"]).fit_transform(X)
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
