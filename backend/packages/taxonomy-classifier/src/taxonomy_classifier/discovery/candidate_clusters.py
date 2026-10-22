"""Candidate clusters from the clustering grid (code only, no LLM).

stability across seeds of the same setting -> size/diversity screens -> de-duplication across settings
-> code-derived parent/child links.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stability import best_match, jac


def candidates(runs, X, meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["clustering"]; st = c["stability"]; seeds = c["seeds"]; ref = seeds[0]
    doc = meta.document_id.to_numpy(object); comp = meta.group.to_numpy(object)
    out = []
    for mcs, ms in c["hdbscan_settings"]:
        lab = runs[(mcs, ms, ref)]
        for k in sorted(set(lab) - {-1}):
            m = np.where(lab == k)[0]
            js, ovs = [], []
            for s in seeds[1:]:
                best_j, best_o = best_match(m, runs[(mcs, ms, s)])
                js.append(best_j); ovs.append(best_o)
            rec = 1 + sum(j >= st["match_jaccard"] for j in js)
            dshare = pd.Series(doc[m]).value_counts(normalize=True)
            cshare = pd.Series(comp[m]).value_counts(normalize=True)
            cen = X[m].mean(0); cen /= np.linalg.norm(cen)
            out.append({"candidate_key": f"{mcs}/{ms}#{k}", "min_cluster_size": mcs, "min_samples": ms, "members": m,
                        "member_count": len(m), "documents": int(dshare.size), "groups": int(cshare.size),
                        "largest_document_share": round(float(dshare.iloc[0]), 3), "largest_group": cshare.index[0],
                        "largest_group_share": round(float(cshare.iloc[0]), 3), "seed_recurrence": int(rec),
                        "mean_jaccard": round(float(np.mean(js)), 3), "mean_overlap": round(float(np.mean(ovs)), 3),
                        "cohesion": round(float((X[m] @ cen).mean()), 4)})
    df = pd.DataFrame(out)
    sc = c["screens"]
    df["stable"] = (df.seed_recurrence >= st["min_seed_recurrence"]) & (df.mean_overlap >= st["min_mean_overlap"])
    df["passes_screens"] = (df.member_count >= sc["min_chunks"]) & (df.documents >= sc["min_documents"]) & (df.largest_document_share <= sc["max_document_share"])
    pool = df[df.stable & df.passes_screens].sort_values(["seed_recurrence", "mean_jaccard", "member_count"], ascending=False)
    reps, rep_of = [], {}
    for i, r in pool.iterrows():
        dup = next((j for j in reps if jac(r.members, df.at[j, "members"]) >= c["dedup_jaccard"]), None)
        if dup is None:
            reps.append(i); rep_of[i] = i
        else:
            rep_of[i] = dup
    df["representative_of"] = df.index.map(lambda i: df.at[rep_of[i], "candidate_key"] if i in rep_of else "")
    df["is_candidate"] = df.index.isin(reps)
    cand = df[df.is_candidate].copy().sort_values(["seed_recurrence", "mean_jaccard", "member_count"], ascending=False)
    cand["cluster_id"] = [f"CL{i:04d}" for i in range(len(cand))]
    df["cluster_id"] = df.index.map(lambda i: cand.at[i, "cluster_id"] if i in cand.index else "")
    df["representative_cluster_id"] = df.representative_of.map(dict(zip(cand.candidate_key, cand.cluster_id))).fillna("")
    return df, cand


def hierarchy(cand: pd.DataFrame, cfg: dict) -> dict:
    """Code-derived parent links among candidates: child sits >= min_overlap inside a parent >= min_size_ratio x larger;
    the smallest qualifying parent is chosen."""
    h = cfg["clustering"]["hierarchy"]
    sets = {r.cluster_id: set(r.members) for r in cand.itertuples()}
    size = {k: len(v) for k, v in sets.items()}
    parent = {}
    for a, A in sets.items():
        ps = [b for b, Bs in sets.items() if b != a and size[b] >= h["min_size_ratio"] * size[a] and len(A & Bs) >= h["min_overlap"] * size[a]]
        if ps:
            parent[a] = min(ps, key=lambda b: size[b])
    return parent


