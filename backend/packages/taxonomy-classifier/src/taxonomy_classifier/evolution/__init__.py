"""Wave evolution: `run_wave` (taxonomy vN -> labels + vN+1) and explicit status changes (`promote`)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from ..assignment.acceptance import CalibratedAcceptance, LearnedAcceptance, make_policy
from ..contract import to_contract
from ..integration.aggregation import aggregate
from ..schemas import parse_chunks
from ..taxonomy import artifacts
from .wave import BYPASS, WaveResult, _Working, novelty_step, write_version

POOL_FIELDS = ("taxonomy_chunk_id", "chunk_id", "document_id", "group", "text", "text_sha256", "token_count", "content_form")


def run_wave(taxonomy, cfg: dict, airweave_chunks, store, wave: int, new_version: str, reviewer, structural_reviewer,
             pool: list[dict] | None = None, embed_transport=None, log=print) -> WaveResult:
    """One wave against taxonomy vN (`taxonomy`). Writes vN+1 into `store` only if categories were added."""
    from ..integration.airweave import TaxonomyClassifier
    chunks = parse_chunks(airweave_chunks)
    clf = TaxonomyClassifier(taxonomy, cfg, embed_transport)
    res = clf.classify(chunks)
    src_ingest = "LEARNED_ACCEPTANCE" if clf.policy.name == "learned" else "HYBRID_RRF"
    labels = {}
    for t in res.child_trace:
        if t["status"] == "ASSIGNED":
            labels[t["taxonomy_chunk_id"]] = {"category_id": t["category_ids"][0], "source": src_ingest, "wave": wave,
                                              "taxonomy_version": taxonomy.version, "prob": t["scores"][0] if t["scores"] else None}
    seen, pool_all = set(), []
    for r in list(pool or []) + res.unresolved_children:
        if r["taxonomy_chunk_id"] not in seen and r["taxonomy_chunk_id"] not in labels:
            seen.add(r["taxonomy_chunk_id"]); pool_all.append({k: r.get(k) for k in POOL_FIELDS})
    pm = pd.DataFrame(pool_all, columns=list(POOL_FIELDS))
    Xp = clf.embedder.embed(list(pm.text)) if len(pm) else np.zeros((0, cfg["embedding"]["dimensions"]), np.float32)
    Bp = taxonomy.bm25.transform(list(pm.text)) if len(pm) else None
    rep = copy.deepcopy(clf.policy.rep if isinstance(clf.policy, LearnedAcceptance) else artifacts.load_representation(taxonomy))
    work = _Working(taxonomy, rep)
    wpolicy = LearnedAcceptance(taxonomy, clf.policy.model, clf.policy.probation, rep=work.rep) if isinstance(clf.policy, LearnedAcceptance) else None
    if wpolicy is not None:
        wpolicy.parent = {c["category_id"]: c.get("parent_category_id") for c in work.final["categories"]}
    decisions, assign = novelty_step(work, wpolicy, cfg, wave, pm, Xp, Bp, reviewer, structural_reviewer, log) if len(pm) else (pd.DataFrame(), {})
    final_tax, out_dir = taxonomy, None
    if work.added:
        out_dir = write_version(work, taxonomy, store, new_version, wave, decisions)
        final_tax = artifacts.load(out_dir)
    for i, (cid, src, rid) in assign.items():
        labels[pm.taxonomy_chunk_id.iat[i]] = {"category_id": cid, "source": src, "wave": wave, "residual_cluster": rid,
                                               "taxonomy_version": final_tax.version}
    rest = [i for i in range(len(pm)) if pm.taxonomy_chunk_id.iat[i] not in labels]
    revisit = 0
    if work.added and rest and cfg.get("evolution", {}).get("revisit_pool_after_change", True):
        pol = make_policy(final_tax, cfg)
        idx = np.array(rest)
        r2 = pol.assign(Xp[idx], Bp[idx], np.zeros(len(idx), bool), list(pm.taxonomy_chunk_id.iloc[idx]),
                        tokens=list(pm.token_count.iloc[idx]), content_forms=list(pm.content_form.iloc[idx]))
        for i, r in zip(idx, r2):
            if r["status"] == "ASSIGNED":
                labels[pm.taxonomy_chunk_id.iat[i]] = {"category_id": r["direct"][0], "source": f"{src_ingest}_REVISIT", "wave": wave,
                                                       "taxonomy_version": final_tax.version, "prob": r["scores"][0] if r["scores"] else None}
                revisit += 1
    new_pool = [r for r in pool_all if r["taxonomy_chunk_id"] not in labels]
    if out_dir is not None:
        with open(out_dir / "evolution" / "unresolved_pool.jsonl", "w") as f:
            for r in new_pool:
                f.write(json.dumps(r) + "\n")
    # contract records for this wave's chunks with the final labels (ingest + novelty + revisit)
    kids = []
    for c in res.children:
        lab = labels.get(c["taxonomy_chunk_id"])
        st = c["status"] if c["status"] in (BYPASS, "REJECT_NONSEMANTIC") else ("ASSIGNED" if lab else "UNRESOLVED")
        kids.append({**c, "status": st, "topics": [{"category_id": lab["category_id"], "source": lab["source"], "score": lab.get("prob")}] if lab else []})
    cfg_agg = {**cfg, "content_form": taxonomy.final.get("content_form", cfg.get("content_form", {}))}
    records = [to_contract(e, final_tax) for e in aggregate(chunks, kids, final_tax, cfg_agg)]
    sub = [c for c in res.children if c["status"] not in (BYPASS, "REJECT_NONSEMANTIC")]
    summary = {"wave": wave, "base_version": taxonomy.version, "new_version": final_tax.version if work.added else None,
               "children": len(res.children), "substantive": len(sub), "assigned_at_ingest": sum(1 for c in sub if c["topics"]),
               "categorised_after_wave": sum(1 for c in sub if c["taxonomy_chunk_id"] in labels),
               "pool_before_novelty": len(pm), "pool_after": len(new_pool), "categories_added": [c["category_id"] for c in work.added],
               "decisions": dict(decisions.decision.value_counts()) if len(decisions) else {}, "revisit_resolved": revisit,
               "acceptance": res.provenance}
    return WaveResult(taxonomy.taxonomy_id, taxonomy.version, final_tax.version if work.added else None, out_dir, labels, decisions, new_pool,
                      records, summary)


__all__ = ["run_wave", "WaveResult", "CalibratedAcceptance", "LearnedAcceptance"]
