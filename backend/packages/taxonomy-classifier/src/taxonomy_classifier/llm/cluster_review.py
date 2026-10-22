"""API-driven semantic review of anonymous candidate clusters.

For each candidate (deterministic order): build payload -> LLM API (cluster_semantic_review_v1) -> schema-validated JSON
-> saved call. The only taxonomy context ever shown is categories ACCEPTED EARLIER IN THIS RUN whose centroid cosine
to the candidate is >= the configured threshold (duplicate check). No historical names/labels are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .client import LLMClient


def representatives(X, members, n_central, n_diverse):
    m = np.asarray(members)
    cen = X[m].mean(0); cen /= np.linalg.norm(cen)
    sim = X[m] @ cen
    order = np.argsort(-sim)
    central = list(m[order[:n_central]])
    chosen = list(central)
    rest = [i for i in m if i not in chosen]
    diverse = []
    while rest and len(diverse) < n_diverse:          # farthest-point (max-min cosine distance) from already chosen
        d = 1 - X[rest] @ X[chosen].T
        j = int(np.argmax(d.min(1)))
        diverse.append(rest[j]); chosen.append(rest[j]); rest.pop(j)
    return central, diverse


def dist(series, n=6):
    vc = series.value_counts(normalize=True).head(n)
    return ", ".join(f"{k} {v:.0%}" for k, v in vc.items())


def review_all(cand: pd.DataFrame, X, meta: pd.DataFrame, terms: dict, cfg: dict, run_dir, ledger, log=print, transport=None) -> pd.DataFrame:
    sr = cfg["semantic_review"]
    cli = LLMClient(cfg, run_dir, ledger, "cluster_semantic_review", sr["prompt_version"], sr["reasoning_effort"], sr["max_output_tokens"], transport)
    text = meta.text.to_numpy(object)
    accepted = []          # (cluster_id, name, description, centroid, member set)
    rows = []
    n_seeds = len(cfg["clustering"]["seeds"])
    for r in cand.itertuples():
        m = np.asarray(r.members)
        cen = X[m].mean(0); cen /= np.linalg.norm(cen)
        sims = sorted(((float(cen @ a[3]), a) for a in accepted), key=lambda t: -t[0])
        sims = [(s, a) for s, a in sims if s >= sr["duplicate_context"]["centroid_cosine"]][: sr["duplicate_context"]["max_candidates"]]
        similar = "\n".join(f"{a[0]} | {a[1]} | {a[2]} | {s:.3f} | {len(set(m) & a[4]) / len(m):.0%}" for s, a in sims) or "(none)"
        central, diverse = representatives(X, m, sr["central_chunks"], sr["diverse_chunks"])
        chunks = "\n\n".join([f"[C{i+1}] ({meta.group.iat[k]}, {meta.doc_type.iat[k]}) " + text[k][: sr["chars_per_chunk"]] for i, k in enumerate(central)] +
                             [f"[D{i+1}] ({meta.group.iat[k]}, {meta.doc_type.iat[k]}) " + text[k][: sr["chars_per_chunk"]] for i, k in enumerate(diverse)])
        sub = meta.iloc[m]
        user = cli.render(cluster_id=r.cluster_id, member_count=len(m), document_count=sub.document_id.nunique(), company_count=sub.group.nunique(),
                          company_distribution=dist(sub.group), sector_distribution=dist(sub.segment), form_distribution=dist(sub.doc_type),
                          seed_recurrence=r.seed_recurrence, n_seeds=n_seeds, mean_jaccard=r.mean_jaccard, min_cluster_size=r.min_cluster_size,
                          min_samples=r.min_samples, top_terms=", ".join(terms[r.cluster_id][: sr["top_terms"]]), similar_categories=similar, chunks=chunks)
        rec = cli.call(user, r.cluster_id, {"representative_taxonomy_chunk_ids": [meta.taxonomy_chunk_id.iat[k] for k in central + diverse],
                                            "similar_categories_shown": [a[0] for _, a in sims]})
        p = rec["parsed"] or {}
        dec = p.get("decision") if rec["status"] == "OK" else f"REVIEW_{rec['status']}"
        if dec == "REJECT_DUPLICATE" and p.get("duplicate_of") not in [a[0] for _, a in sims]:
            dec = "REJECT_DUPLICATE_INVALID_TARGET"       # deterministic rule: duplicate must reference a listed category
        if dec == "ACCEPT":
            accepted.append((r.cluster_id, p["category_name"], p["description"], cen, set(m)))
        rows.append({"cluster_id": r.cluster_id, "request_id": rec["request_id"], "status": rec["status"], "cache_hit": rec["cache_hit"],
                     "decision": dec, "coherent": p.get("coherent"), "category_name": p.get("category_name"), "description": p.get("description"),
                     "scope": p.get("scope"), "industry": p.get("industry"), "company": p.get("company"),
                     "semantic_or_structural": p.get("semantic_or_structural"), "duplicate_of": p.get("duplicate_of"),
                     "confidence": p.get("confidence"), "reason": p.get("reason"), "similar_shown": ";".join(a[0] for _, a in sims),
                     "representative_chunk_ids": ";".join(meta.taxonomy_chunk_id.iat[k] for k in central + diverse)})
        if len(rows) % 20 == 0:
            log(f"reviewed {len(rows)}/{len(cand)} (accepted {len(accepted)}; spent ${ledger.data['total_usd']:.3f})")
    return pd.DataFrame(rows)
