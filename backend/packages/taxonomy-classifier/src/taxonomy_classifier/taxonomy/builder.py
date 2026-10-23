"""Build the fine-grained taxonomy ONLY from logged semantic-review responses.

ACCEPT -> category (name/description/scope copied verbatim from the parsed API response; COMPANY scope kept and flagged)
REJECT_DUPLICATE -> merge recommendation into the referenced category (no category created, nothing merged silently)
REJECT_ADMINISTRATIVE / REJECT_STRUCTURAL -> reject prototypes for assignment (non-semantic evidence)
everything else -> not a category (recorded)
Parent links: code-derived containment links between candidates, restricted to accepted categories (walk up).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build(cand: pd.DataFrame, reviews: pd.DataFrame, parent_links: dict, X, meta: pd.DataFrame, terms: dict, sparse_profile_fn) -> dict:
    rv = reviews.set_index("cluster_id")
    acc = [c for c in cand.cluster_id if rv.at[c, "decision"] == "ACCEPT"]
    cat_id = {c: f"CAT{i+1:03d}" for i, c in enumerate(acc)}
    mem = dict(zip(cand.cluster_id, cand.members))

    def accepted_parent(c):
        p = parent_links.get(c)
        while p is not None and p not in cat_id:
            p = parent_links.get(p)
        return p
    cats = []
    for c in acc:
        m = np.asarray(mem[c])
        cen = X[m].mean(0); cen /= np.linalg.norm(cen)
        sub = meta.iloc[m]
        r = rv.loc[c]
        p = accepted_parent(c)
        cats.append({"category_id": cat_id[c], "name": r.category_name, "description": r.description, "scope": r.scope,
                     "industry": r.industry, "company": r.company, "confidence": r.confidence,
                     "parent_category_id": cat_id[p] if p else None, "source_cluster_id": c,
                     "source_review_call_id": r.request_id, "member_taxonomy_chunk_ids": list(meta.taxonomy_chunk_id.iloc[m]),
                     "member_airweave_chunk_ids": sorted(set(sub.chunk_id)), "member_count": int(len(m)),
                     "document_count": int(sub.document_id.nunique()), "group_count": int(sub.group.nunique()),
                     "groups": sorted(set(map(str, sub.group))), "top_terms": terms[c], "representative_chunk_ids": r.representative_chunk_ids.split(";"),
                     "dense_prototype": [round(float(v), 6) for v in cen], "sparse_profile": sparse_profile_fn(m)})
    merges = [{"cluster_id": c, "recommend_merge_into": next((cat_id.get(x) for x in [rv.at[c, "duplicate_of"]] if x in cat_id), None),
               "duplicate_of_cluster": rv.at[c, "duplicate_of"], "request_id": rv.at[c, "request_id"], "reason": rv.at[c, "reason"]}
              for c in cand.cluster_id if rv.at[c, "decision"] == "REJECT_DUPLICATE"]
    reject_groups = [np.asarray(mem[c]) for c in cand.cluster_id if rv.at[c, "decision"] in ("REJECT_ADMINISTRATIVE", "REJECT_STRUCTURAL")]
    return {"categories": cats, "merge_recommendations": merges, "reject_groups": reject_groups, "cluster_to_category": cat_id}
