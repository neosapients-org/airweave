"""Versioned wave evolution: taxonomy vN + incoming documents -> labels + decisions -> taxonomy vN+1 (vN never modified).

    classify the wave against vN (acceptance policy)            -> labels LEARNED_ACCEPTANCE / HYBRID_RRF
    unresolved substantive children join the pool (carried across waves)
    residual clustering of the pool (UMAP x HDBSCAN, multi-seed)
    per recurring cluster:
      recurrence gate           (>= novelty.min_cluster_size_for_review children, >= creation.min_documents documents)
      code precheck             centroid cosine >= 0.85 to its nearest category, members' top-1 agree (share >= 0.60) -> MAP_EXISTING
      otherwise cluster review  (pluggable reviewer; LLM in production, deterministic fakes in tests)
      MAP_EXISTING              members verified in their own dense top-k; members mapped into a non-MATURE category must also
                                pass that category's probation gate (learned policy); retired categories accept nobody
      CREATE_NEW                coherent, does not fit, allowed scope, >= min groups; members verified against the new
                                centroid; recurrence re-checked on verified members; structural filter must say SEMANTIC_TOPIC;
                                the category starts PROBATIONARY
      STRUCTURAL_OR_ADMIN / KEEP_UNRESOLVED otherwise
    categories created in this wave are visible to later clusters of the same wave (validated research behaviour)
    if the taxonomy changed: the remaining pool is re-classified against vN+1 (LEARNED_ACCEPTANCE_REVISIT)
    vN+1 is written as a NEW version (store), recording its parent version, the wave and every decision

Port of the validated research engine (taxonomy growth scenario A with the category-maturity arm D policy); see
docs/VALIDATED_BASELINE_SPEC.md. Promotion is NOT automatic (not validated); use evolution.promote.promote().
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..assignment.acceptance import LearnedAcceptance
from ..assignment.rrf import ranks
from ..taxonomy import artifacts, maturity
from ..taxonomy.representation import category_stats

BYPASS = "BYPASS_ADMINISTRATIVE"


@dataclass
class WaveResult:
    taxonomy_id: str
    base_version: str
    new_version: str | None            # None when the wave changed nothing in the taxonomy
    new_taxonomy_path: Path | None
    labels: dict                       # taxonomy_chunk_id -> {category_id, source, wave, taxonomy_version, prob?}
    decisions: pd.DataFrame
    pool: list[dict]                   # unresolved children carried to the next wave
    records: list[dict] = field(default_factory=list)   # contract records for this wave's Airweave chunks (final labels)
    summary: dict = field(default_factory=dict)


class _Working:
    """Mutable working copy of the category set during one wave (never written back to vN)."""

    def __init__(self, tax, rep):
        self.final = copy.deepcopy(tax.final)
        self.rep = rep
        a = tax.final["assignment"]
        P = len(a["prototype_category_ids"])
        self.Dp_pos, self.Dp_rej = tax.assigner.Dp[:P].copy(), tax.assigner.Dp[P:].copy()
        self.Sp_pos, self.Sp_rej = tax.assigner.Sp[:P].tocsr(), tax.assigner.Sp[P:].tocsr()
        self.added: list[dict] = []

    @property
    def ids(self):
        return self.rep.ids


def _status(work, cid):
    return work.rep.statuses[work.rep.index(cid)]


def _nearest(work, cen, prof, n_cand):
    ds = work.rep.centroids @ cen
    ss = np.asarray(work.rep.sparse @ prof).ravel()
    k = work.final["assignment"]["rrf_k"]                # the taxonomy's own RRF constant, as its classification uses
    r = 1 / (k + ranks(ds[None])[0]) + 1 / (k + ranks(ss[None])[0])
    return [work.ids[j] for j in np.argsort(-r)[:n_cand]], ds


def _add_category(work, cfg, wave, rid, review, idx, X, B, meta):
    cen = X[idx].mean(0); cen /= np.linalg.norm(cen)
    v = np.asarray(B[idx].mean(0)).ravel(); keep = np.argsort(-v)[: cfg["assignment"]["profile_terms"]]
    pv = np.zeros_like(v); pv[keep] = v[keep]; pv /= np.linalg.norm(pv) or 1
    ev = cfg.get("evolution", {})
    prefix = ev.get("new_category_prefix", "CAT_NEW_")
    n, cid = sum(1 for c in work.final["categories"] if c.get("origin", "BUILD").startswith("WAVE")) + 1, None
    while cid is None or cid in work.ids:
        cid = f"{prefix}{n:03d}"; n += 1
    status = ev.get("new_category_status", maturity.PROBATIONARY)
    stats = category_stats(X[idx], cen, meta.group.to_numpy(object)[idx], meta.document_id.to_numpy(object)[idx], False, False)
    cat = {"category_id": cid, "name": review["new_category_name"], "description": review.get("description"), "scope": review.get("scope"),
           "industry": None, "company": None, "parent_category_id": None, "broad_family_id": None, "broad_family": None,
           "status": status, "is_sink": False, "stats": stats, "origin": f"WAVE_{wave}", "member_count": int(len(idx)),
           "created": {"wave": wave, "source_residual_cluster": rid, "source_review_call_id": review.get("call_id"),
                       "member_taxonomy_chunk_ids": list(meta.taxonomy_chunk_id.iloc[idx])}}
    r = work.rep
    r.ids.append(cid); r.centroids = np.vstack([r.centroids, cen[None]]).astype(np.float32)
    r.prototypes.append(cen[None].astype(np.float32)); r.exemplars.append(X[idx].astype(np.float32))
    r.sparse = sp.vstack([r.sparse, sp.csr_matrix(pv[None])]).tocsr(); r.stats.append(stats)
    r.scopes.append(cat["scope"]); r.families.append(None); r.statuses.append(status)
    work.final["categories"].append(cat); work.added.append(cat)
    # calibrated-policy arrays: the new category becomes a positive prototype at the taxonomy-wide dense floor
    work.Dp_pos = np.vstack([work.Dp_pos, cen[None]]).astype(np.float32)
    work.Sp_pos = sp.vstack([work.Sp_pos, sp.csr_matrix(pv[None])]).tocsr()
    a = work.final["assignment"]
    a["prototype_category_ids"].append(cid); a["parent"][cid] = None
    a["category_floors"][cid] = float(work.final.get("calibration", {}).get("dense_floor", 0.95))
    return cat


def novelty_step(work, policy, cfg, wave, pool_meta, X, B, reviewer, structural, log=print):
    """Decide every recurring residual cluster of the pool. Returns (decisions DataFrame, {pool row: (category_id, source, rid)})."""
    from ..discovery.clustering import fit_terms, top_terms
    from ..llm.cluster_review import dist, representatives
    from ..novelty.maintenance import residual_clusters
    from sklearn.feature_extraction.text import CountVectorizer

    n_cfg, ev = cfg["novelty"], cfg.get("evolution", {})
    pre, cr = ev.get("precheck", {"centroid_cosine_min": 0.85, "member_top1_share_min": 0.60}), ev.get("creation", {})
    min_docs, min_groups = cr.get("min_documents", n_cfg["min_documents"]), cr.get("min_groups", n_cfg["min_companies_new_category"])
    scopes, K = tuple(cr.get("scopes", ("GLOBAL", "INDUSTRY"))), n_cfg["per_chunk_verification"]["k"]
    rows, assign = [], {}
    if len(pool_meta) < ev.get("min_pool_for_clustering", 20):
        log(f"pool {len(pool_meta)} < {ev.get('min_pool_for_clustering', 20)}: no residual clustering"); return pd.DataFrame(rows), assign
    clusters = residual_clusters(X, cfg)
    cv, TF = fit_terms(CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min(3, max(1, len(pool_meta) // 50)),
                                       token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"), pool_meta.text)
    for ci, cl in enumerate(clusters):
        rid = f"W{wave}-R{ci + 1:03d}"; m = np.asarray(cl["members"]); sub = pool_meta.iloc[m]
        cen = X[m].mean(0); cen /= np.linalg.norm(cen); prof = np.asarray(B[m].mean(0)).ravel(); prof /= np.linalg.norm(prof) or 1
        near, ds = _nearest(work, cen, prof, n_cfg["candidate_categories"])
        cmap = {c["category_id"]: c for c in work.final["categories"]}
        rec = {"wave": wave, "residual_cluster_id": rid, "members": len(m), "documents": sub.document_id.nunique(), "groups": sub.group.nunique(),
               "nearest_categories": ";".join(near[:5]), "nearest_cosine": round(float(ds.max()), 4),
               "member_taxonomy_chunk_ids": ";".join(sub.taxonomy_chunk_id)}
        if len(m) < n_cfg["min_cluster_size_for_review"] or rec["documents"] < min_docs:
            rows.append({**rec, "decision": "KEEP_UNRESOLVED", "decided_by": "CODE_RECURRENCE_GATE"}); continue
        target, decided, p = None, None, {}
        C = work.rep.centroids
        top1 = np.argmax(X[m] @ C.T, 1); t1c = Counter(top1.tolist()).most_common(1)[0]
        if ds.max() >= pre["centroid_cosine_min"] and t1c[0] == int(np.argmax(ds)) and t1c[1] / len(m) >= pre["member_top1_share_min"]:
            target, decided = work.ids[int(np.argmax(ds))], "CODE_PRECHECK"
        else:
            central, diverse = representatives(X, m, n_cfg["central_chunks"], n_cfg["diverse_chunks"])
            chunks = "\n\n".join(f"[{'C' if k in central else 'D'}{i + 1}] ({pool_meta.group.iat[k]}) " + pool_meta.text.iat[k][: n_cfg["chars_per_chunk"]]
                                 for i, k in enumerate(central + diverse))
            ctx = {"cluster_id": rid, "member_count": len(m), "document_count": rec["documents"], "group_count": rec["groups"],
                   "group_distribution": dist(sub.group), "top_terms": top_terms(cv, TF, {"x": m}, 12)["x"],
                   "candidates": [{"category_id": c, "name": cmap[c]["name"], "description": cmap[c].get("description")} for c in near],
                   "chunks": chunks, "member_taxonomy_chunk_ids": list(sub.taxonomy_chunk_id)}
            out = reviewer.review(ctx)
            p = out.get("parsed") or {}
            rec.update({"review_call_id": out.get("call_id"), **{f"review_{k}": v for k, v in p.items()}})
            if out.get("status", "OK") != "OK":
                rows.append({**rec, "decision": "KEEP_UNRESOLVED", "decided_by": f"REVIEW_{out.get('status')}"}); continue
            decided = "REVIEW"
            if p.get("fits_existing") and p.get("existing_category_id") in near:
                target = p["existing_category_id"]
        if target:
            tj = work.ids.index(target)
            own = np.argsort(-(X[m] @ C.T), 1)[:, :K]
            ok = [int(i) for i, row in zip(m, own) if tj in row]
            st = _status(work, target)
            if (st != maturity.MATURE and ok and isinstance(policy, LearnedAcceptance)
                    and ev.get("map_into_young_categories_requires_gate", True)):
                okk = np.array(ok)
                passed = policy.passes_gate(X[okk], B[okk], pool_meta.token_count.to_numpy()[okk], list(pool_meta.content_form.iloc[okk]), target)
                ok = [i for i, g in zip(ok, passed) if g]
            if st in maturity.RETIRED:
                ok = []
            for i in ok:
                assign[i] = (target, "NOVELTY_VERIFIED_MAPPING", rid)
            rows.append({**rec, "decision": "MAP_EXISTING", "decided_by": decided, "target_category_id": target, "target_status": st,
                         "members_verified": len(ok), "members_rejected": len(m) - len(ok)}); continue
        if p.get("scope") == "NONE":
            rows.append({**rec, "decision": "STRUCTURAL_OR_ADMIN", "decided_by": decided}); continue
        if p.get("coherent") and not p.get("fits_existing") and p.get("scope") in scopes and p.get("new_category_name") and rec["groups"] >= min_groups:
            C2 = np.vstack([C, cen[None]]); top = np.argsort(-(X[m] @ C2.T), 1)[:, :K]
            ok = np.array([int(i) for i, row in zip(m, top) if len(C) in row], dtype=int)
            okd = pool_meta.document_id.iloc[ok].nunique() if len(ok) else 0; okg = pool_meta.group.iloc[ok].nunique() if len(ok) else 0
            if len(ok) >= n_cfg["min_cluster_size_for_review"] and okd >= min_docs and okg >= min_groups:
                sf = ev.get("structural_filter", {"enabled": True})
                if sf.get("enabled", True):
                    central, diverse = representatives(X, m, 4, 2)
                    s_out = structural.review({"name": p["new_category_name"], "description": p.get("description"),
                                               "terms": top_terms(cv, TF, {"x": m}, 12)["x"],
                                               "chunks": "\n\n".join(f"({pool_meta.group.iat[k]}) {pool_meta.text.iat[k][:700]}" for k in central + diverse),
                                               "cluster_id": rid, "member_taxonomy_chunk_ids": list(sub.taxonomy_chunk_id)})
                    rec.update({"structural_class": s_out.get("classification"), "structural_call_id": s_out.get("call_id")})
                    if s_out.get("classification") != "SEMANTIC_TOPIC":
                        rows.append({**rec, "decision": "STRUCTURAL_OR_ADMIN", "decided_by": f"STRUCTURAL_FILTER:{s_out.get('classification')}",
                                     "proposed_name": p["new_category_name"]}); continue
                cat = _add_category(work, cfg, wave, rid, {**p, "call_id": rec.get("review_call_id")}, ok, X, B, pool_meta)
                for i in ok:
                    assign[int(i)] = (cat["category_id"], "NOVELTY_NEW_CATEGORY", rid)
                rows.append({**rec, "decision": "CREATE_NEW", "decided_by": decided, "target_category_id": cat["category_id"],
                             "target_name": cat["name"], "target_status": cat["status"], "members_verified": len(ok)}); continue
            rows.append({**rec, "decision": "KEEP_UNRESOLVED", "decided_by": "CODE_VERIFIED_RECURRENCE_GATE"}); continue
        why = "NOT_COHERENT" if not p.get("coherent") else "GROUP_SCOPE" if p.get("scope") not in scopes else "RECURRENCE_GATE"
        rows.append({**rec, "decision": "KEEP_UNRESOLVED", "decided_by": f"REVIEW+{why}"})
    log(f"novelty: pool {len(pool_meta)} clusters {len(clusters)} {dict(Counter(r['decision'] for r in rows))} labelled {len(assign)}")
    return pd.DataFrame(rows), assign


def write_version(work, base_tax, store_or_dir, new_version: str, wave: int, decisions: pd.DataFrame) -> Path:
    """Write the working category set as a NEW immutable taxonomy version (never touches base_tax.path)."""
    from ..assignment.classifier import FrozenAssigner
    final = work.final
    final.update({"taxonomy_version": new_version, "created_at": datetime.now(timezone.utc).isoformat(),
                  "derived_from": {"taxonomy_id": base_tax.taxonomy_id, "taxonomy_version": base_tax.version,
                                   "final_taxonomy_sha256": artifacts._sha(base_tax.path / "final_taxonomy.json")},
                  "evolution": {"wave": wave, "categories_added": [c["category_id"] for c in work.added],
                                "decisions": dict(Counter(decisions.decision)) if len(decisions) else {}}})
    final.pop("artifacts", None)
    out = store_or_dir.path(base_tax.taxonomy_id, new_version) if hasattr(store_or_dir, "path") else Path(store_or_dir)
    if out.exists():
        raise FileExistsError(f"{out} already exists; taxonomy versions are immutable")
    a = final["assignment"]
    Dp = np.vstack([work.Dp_pos, work.Dp_rej]).astype(np.float32); Sp = sp.vstack([work.Sp_pos, work.Sp_rej]).tocsr()
    pos = a["prototype_category_ids"]
    fa = FrozenAssigner(pos, a["parent"], Dp, Sp, len(work.Dp_rej), [a["category_floors"][c] for c in pos], a["tau_rrf"], a["sibling_delta"],
                        a["window"], a["rrf_k"])
    body = {k: v for k, v in final.items() if k != "assignment"}
    artifacts.save(out, body, fa, base_tax.bm25, work.rep.centroids, work.rep.sparse, work.rep.prototypes, work.rep.exemplars)
    ev = out / "evolution"; ev.mkdir(exist_ok=True)
    decisions.to_csv(ev / "decisions.csv", index=False)
    return out
