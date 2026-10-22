"""Novelty maintenance (separate, periodic job): unresolved pool -> residual clustering (code) -> whole-taxonomy candidate search (code)
-> LLM novelty review (cluster level only) -> per-chunk verification (code) -> mappings / new-category proposals.

Per-chunk verification rule (config novelty.per_chunk_verification): a residual chunk is mapped to existing category X
only if X is within the chunk's OWN dense top-k over ALL categories. Chunks failing the rule stay unresolved.
New categories require: coherent, fits_existing=false, scope GLOBAL/INDUSTRY, >= min_cluster_size_for_review chunks,
>= min_documents documents, >= min_companies_new_category source groups. Their name/description come verbatim from the API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..assignment.rrf import ranks
from ..discovery.stability import jac
from ..llm import novelty_review
from ..llm.cluster_review import representatives


def residual_clusters(Xp, cfg):
    import hdbscan
    import umap
    n = cfg["novelty"]
    emb = {s: umap.UMAP(random_state=s, **n["umap"]).fit_transform(Xp) for s in n["seeds"]}
    cands = []
    for mcs, ms in n["hdbscan_settings"]:
        runs = {s: hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="leaf").fit_predict(emb[s]) for s in n["seeds"]}
        lab = runs[n["seeds"][0]]
        for k in sorted(set(lab) - {-1}):
            m = np.where(lab == k)[0]
            js = [max([jac(m, np.where(runs[s] == k2)[0]) for k2 in set(runs[s][m]) - {-1}] or [0]) for s in n["seeds"][1:]]
            cands.append({"setting": f"{mcs}/{ms}", "members": m, "size": len(m), "mean_jaccard": float(np.mean(js)), "stable": min(js) >= 0.5})
    df = pd.DataFrame(cands)
    if df.empty:
        return []
    st = df[df.stable].sort_values(["mean_jaccard", "size"], ascending=False)
    taken = np.zeros(len(Xp), bool); out = []
    for _, r in st.iterrows():
        if taken[r.members].mean() > 0.3:
            continue
        m = r.members[~taken[r.members]]
        if len(m) < 5:
            continue
        taken[m] = True
        out.append({"setting": r.setting, "mean_jaccard": round(r.mean_jaccard, 3), "members": m})
    return out


def run(pool_idx, X, B, meta, cats, cat_C, cat_S, cfg, run_dir, ledger, terms_fn, log=print, transport=None):
    """pool_idx: indices (into X/meta) of unresolved substantive chunks. cat_C: dense centroids of ALL categories,
    cat_S: sparse profiles of ALL categories (rows aligned with cats)."""
    n = cfg["novelty"]
    ids = [c["category_id"] for c in cats]; name = {c["category_id"]: c["name"] for c in cats}; desc = {c["category_id"]: c["description"] for c in cats}
    Xp = X[pool_idx]
    clusters = residual_clusters(Xp, cfg) if len(pool_idx) >= 20 else []
    cli = novelty_review.client(cfg, run_dir, ledger, transport)
    text = meta.text.to_numpy(object)
    Sd_all = X[pool_idx] @ cat_C.T
    own_top = np.argsort(-Sd_all, 1)[:, : n["per_chunk_verification"]["k"]]
    rows, mappings, newcats = [], [], []
    for ci, cl in enumerate(clusters):
        rid = f"RC{ci+1:03d}"
        m_local = cl["members"]; m = pool_idx[m_local]
        sub = meta.iloc[m]
        rec = {"residual_cluster_id": rid, "setting": cl["setting"], "seed_mean_jaccard": cl["mean_jaccard"], "members": len(m),
               "documents": sub.document_id.nunique(), "groups": sub.group.nunique(), "group_list": ";".join(sorted(set(map(str, sub.group)))),
               "member_taxonomy_chunk_ids": ";".join(sub.taxonomy_chunk_id)}
        if len(m) < n["min_cluster_size_for_review"] or rec["documents"] < n["min_documents"]:
            rows.append({**rec, "resolution": "NOT_REVIEWED_INSUFFICIENT_RECURRENCE"}); continue
        cen = X[m].mean(0); cen /= np.linalg.norm(cen)
        prof = np.asarray(B[m].mean(0)).ravel(); prof /= np.linalg.norm(prof) or 1
        ds = cat_C @ cen; ss = np.asarray(cat_S @ prof).ravel()
        k = cfg["assignment"]["rrf_k"]
        r = 1 / (k + ranks(ds[None])[0]) + 1 / (k + ranks(ss[None])[0])
        cand = [ids[j] for j in np.argsort(-r)[: n["candidate_categories"]]]
        central, diverse = representatives(X, m, n["central_chunks"], n["diverse_chunks"])
        chunks = "\n\n".join(f"[{'C' if k in central else 'D'}{i+1}] ({meta.group.iat[k]}) " + text[k][: n["chars_per_chunk"]] for i, k in enumerate(central + diverse))
        user = novelty_review.payload(cli, rid, sub, terms_fn(m), cand, name, desc, chunks)
        call = cli.call(user, rid, {"candidate_category_ids": cand})
        p = call["parsed"] or {}
        rec.update({"request_id": call["request_id"], "call_status": call["status"], "candidates": ";".join(cand), **{f"llm_{k}": v for k, v in p.items()}})
        if call["status"] != "OK":
            rows.append({**rec, "resolution": f"REVIEW_{call['status']}"}); continue
        if p["fits_existing"] and p.get("existing_category_id") in cand:
            X_id = p["existing_category_id"]; xj = ids.index(X_id)
            ok = [int(i) for i, li in zip(m, m_local) if xj in own_top[li]]
            for i in m:
                mappings.append({"taxonomy_chunk_id": meta.taxonomy_chunk_id.iat[i], "residual_cluster_id": rid, "category_id": X_id,
                                 "verified": int(i) in ok, "rule": f"category in chunk's own dense top-{n['per_chunk_verification']['k']}",
                                 "request_id": call["request_id"]})
            rows.append({**rec, "resolution": "MAPPED_TO_EXISTING", "target_category_id": X_id, "verified_members": len(ok)})
        elif (p["coherent"] and not p["fits_existing"] and p["scope"] in ("GLOBAL", "INDUSTRY") and len(m) >= n["min_cluster_size_for_review"]
              and rec["documents"] >= n["min_documents"] and rec["groups"] >= n["min_companies_new_category"] and p.get("new_category_name")):
            nid = f"NEW{len(newcats)+1:03d}"
            newcats.append({"category_id": nid, "name": p["new_category_name"], "description": p["description"], "scope": p["scope"],
                            "source": "novelty_review", "source_residual_cluster": rid, "source_review_call_id": call["request_id"],
                            "member_taxonomy_chunk_ids": list(sub.taxonomy_chunk_id), "member_count": len(m), "document_count": rec["documents"],
                            "group_count": rec["groups"], "dense_prototype": [round(float(v), 6) for v in cen]})
            rows.append({**rec, "resolution": "NEW_CATEGORY_ACCEPTED", "target_category_id": nid})
        else:
            why = ("NOT_COHERENT" if not p["coherent"] else "NON_SEMANTIC" if p["scope"] == "NONE" else
                   "COMPANY_SPECIFIC" if p["scope"] == "COMPANY" else "INVALID_EXISTING_ID" if p["fits_existing"] else "RECURRENCE_GATE")
            rows.append({**rec, "resolution": f"NOT_ACCEPTED_{why}"})
    return {"clusters": pd.DataFrame(rows), "mappings": pd.DataFrame(mappings), "new_categories": newcats,
            "pool_size": int(len(pool_idx)), "clustered": int(sum(len(c["members"]) for c in clusters))}


def run_maintenance(taxonomy, unresolved: list[dict], cfg: dict, out_dir, log=print, embed_transport=None, llm_transport=None) -> dict:
    """Periodic job over the accumulated novelty pool (records written by classify_airweave_chunks --unresolved-output).

    Writes novelty_residual_clusters.csv, novelty_mappings.csv and taxonomy_maintenance_proposal.json to out_dir.
    The frozen taxonomy is NOT modified: mappings and new categories are proposals for the next taxonomy version.
    """
    from pathlib import Path

    from sklearn.feature_extraction.text import CountVectorizer

    from ..discovery.clustering import top_terms
    from ..embeddings.embedder import Embedder
    from ..llm.costs import CostLedger
    from ..schemas import write_json

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if taxonomy.category_centroids is None or taxonomy.category_sparse is None:
        raise FileNotFoundError("novelty maintenance needs category_centroids.npy and category_sparse_profiles.npz in the taxonomy directory")
    seen, rows = set(), []
    for r in unresolved:                                   # the pool may contain the same child twice across batches
        if r["taxonomy_chunk_id"] not in seen:
            seen.add(r["taxonomy_chunk_id"]); rows.append(r)
    meta = pd.DataFrame([{"taxonomy_chunk_id": r["taxonomy_chunk_id"], "chunk_id": r["chunk_id"], "document_id": r["document_id"],
                          "group": str(r.get("group") or r["document_id"]), "text": r["text"]} for r in rows])
    ledger = CostLedger(cfg["llm"]["max_cost_usd"], out / "cost_ledger.json")
    X = Embedder(cfg, ledger, embed_transport).embed(list(meta.text)) if len(meta) else np.zeros((0, cfg["embedding"]["dimensions"]), np.float32)
    B = taxonomy.bm25.transform(list(meta.text)) if len(meta) else None
    cats = [{"category_id": c["category_id"], "name": c["name"], "description": c["description"]} for c in taxonomy.categories]
    terms_fn = lambda idx: []
    if len(meta) >= 3:
        cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min(3, len(meta)), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        try:
            TF = cv.fit_transform(meta.text)
            terms_fn = lambda idx: top_terms(cv, TF, {"x": idx}, 12)["x"]
        except ValueError:                                 # empty vocabulary
            pass
    res = run(np.arange(len(meta)), X, B, meta, cats, taxonomy.category_centroids, taxonomy.category_sparse, cfg, out, ledger,
              terms_fn, log, llm_transport) if len(meta) else {"clusters": pd.DataFrame(), "mappings": pd.DataFrame(), "new_categories": [],
                                                               "pool_size": 0, "clustered": 0}
    res["clusters"].to_csv(out / "novelty_residual_clusters.csv", index=False)
    res["mappings"].to_csv(out / "novelty_mappings.csv", index=False)
    summary = {"base_taxonomy_version": taxonomy.version, "pool_size": res["pool_size"], "clustered": res["clustered"],
               "residual_clusters": int(len(res["clusters"])),
               "verified_mappings": int(res["mappings"].verified.sum()) if len(res["mappings"]) else 0,
               "new_categories": res["new_categories"], "verification_rule": cfg["novelty"]["per_chunk_verification"],
               "cost_usd": round(ledger.data["total_usd"], 4),
               "note": "Proposal only. The frozen taxonomy is unchanged; promote proposals by rebuilding or re-consolidating."}
    write_json(summary, out / "taxonomy_maintenance_proposal.json")
    log(f"novelty: pool {summary['pool_size']}, residual clusters {summary['residual_clusters']}, "
        f"verified mappings {summary['verified_mappings']}, new categories {len(res['new_categories'])}")
    return summary
