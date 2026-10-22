"""Acceptance policies: which candidate categories a taxonomy child receives. The classify pipeline only calls
`policy.assign(...)`, so a new policy (e.g. multi-label) plugs in without changing the pipeline.

    calibrated  dense + BM25 + RRF with per-taxonomy calibrated floors / tau (FrozenAssigner; the build default)
    learned     dense + BM25 + RRF candidates -> 40 features -> frozen classifier P(correct) -> status gate
                (MATURE: P >= model threshold; PROBATIONARY: validated strict gate) -> top `max_labels` (1) by P

Every policy returns one record per child:
  {taxonomy_chunk_id, status (ASSIGNED | UNRESOLVED | BYPASS_ADMINISTRATIVE | REJECT_NONSEMANTIC), direct, inherited, scores,
   parent_fallback, trace}
Zero labels is a valid outcome (UNRESOLVED): no policy forces a top-1 label.
"""

from __future__ import annotations

import numpy as np

from ..taxonomy import maturity
from . import learned as L

CALIBRATED, LEARNED = "calibrated", "learned"


class CalibratedAcceptance:
    name = CALIBRATED

    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

    def provenance(self) -> dict:
        return {"acceptance_policy": CALIBRATED, "acceptance_model": None, "threshold": {"tau_rrf": self.taxonomy.assigner.tau}}

    def assign(self, X, B, admin, ids, tokens=None, content_forms=None) -> list[dict]:
        out = self.taxonomy.assigner.assign(X, B, admin, ids)
        for r in out:
            r["trace"] = {"rrf_top": r.get("rrf_top", [])[:3]}
        return out


class LearnedAcceptance:
    name = LEARNED

    def __init__(self, taxonomy, model: L.LearnedAcceptanceModel, probation_gate: dict | None = None, rep=None):
        """rep: optional working CategoryRepresentation (wave evolution scores against categories added mid-wave)."""
        from ..taxonomy.artifacts import load_representation
        self.taxonomy, self.model = taxonomy, model
        self.rep = rep if rep is not None else load_representation(taxonomy)
        self.probation = {**maturity.VALIDATED_PROBATION_GATE, **(probation_gate or {})}
        self.parent = {c["category_id"]: c.get("parent_category_id") for c in taxonomy.categories}

    def provenance(self) -> dict:
        return {"acceptance_policy": LEARNED, "acceptance_model": self.model.model_id, "acceptance_model_sha256": self.model.manifest["model_sha256"],
                "threshold": self.model.threshold, "max_labels_per_child": self.model.max_labels, "probation_gate": self.probation}

    def score_pairs(self, X, B, admin, tokens, content_forms):
        """All candidate pairs with features, P(correct), the category status and the gate decision."""
        M = L.score_matrices(self.rep, X, B)
        pf = L.pair_features(self.rep, M, tokens, content_forms, admin)
        if pf.empty:
            pf["p"] = []; pf["status"] = []; pf["gate_ok"] = []; return pf
        pf["p"] = self.model.predict(L.design(pf))
        st = dict(zip(self.rep.ids, self.rep.statuses))
        pf["status"] = pf.category_id.map(st)
        pf["gate_ok"] = [maturity.gate_ok(p, b, d, s, self.model.threshold, self.probation)
                         for p, b, d, s in zip(pf.p, pf.bm25_rank, pf.dense_rank, pf.status)]
        return pf

    def assign(self, X, B, admin, ids, tokens=None, content_forms=None) -> list[dict]:
        n = len(X)
        tokens = tokens if tokens is not None else np.zeros(n, int)
        content_forms = content_forms if content_forms is not None else ["NARRATIVE"] * n
        pf = self.score_pairs(X, B, admin, tokens, content_forms)
        acc = pf[pf.gate_ok.astype(bool)].sort_values("p", ascending=False) if len(pf) else pf
        acc = acc[acc.groupby("k").cumcount() < self.model.max_labels] if len(acc) else acc
        chosen = {}
        for r in acc.itertuples():
            chosen.setdefault(int(r.k), []).append(r)
        best = pf.sort_values("p", ascending=False).drop_duplicates("k").set_index("k") if len(pf) else None
        out = []
        for i in range(n):
            base = {"taxonomy_chunk_id": ids[i], "inherited": [], "parent_fallback": False}
            if admin[i]:
                out.append({**base, "status": "BYPASS_ADMINISTRATIVE", "direct": [], "scores": [], "trace": {}}); continue
            top = best.loc[i] if best is not None and i in best.index else None
            trace = {} if top is None else {"best_candidate": top.category_id, "best_p": round(float(top.p), 4), "best_status": top.status,
                                            "best_bm25_rank": int(top.bm25_rank), "best_dense_rank": int(top.dense_rank), "best_gate_ok": bool(top.gate_ok)}
            if i not in chosen:
                out.append({**base, "status": "UNRESOLVED", "direct": [], "scores": [], "trace": trace}); continue
            rs = chosen[i]
            out.append({**base, "status": "ASSIGNED", "direct": [r.category_id for r in rs], "scores": [round(float(r.p), 6) for r in rs],
                        "inherited": sorted({a for r in rs for a in self._ancestors(r.category_id)}),
                        "trace": {**trace, "accepted_p": [round(float(r.p), 4) for r in rs], "accepted_status": [r.status for r in rs]}})
        return out

    def _ancestors(self, cid):
        out, p, seen = [], self.parent.get(cid), {cid}
        while p and p not in seen:
            seen.add(p); out.append(p); p = self.parent.get(p)
        return out

    def passes_gate(self, X, B, tokens, content_forms, category_id: str) -> np.ndarray:
        """Per child: is `category_id` a candidate that passes its status gate? (novelty mapping into young categories)"""
        pf = self.score_pairs(X, B, np.zeros(len(X), bool), tokens, content_forms)
        ok = np.zeros(len(X), bool)
        if len(pf):
            sel = pf[(pf.category_id == category_id) & pf.gate_ok.astype(bool)]
            ok[sel.k.astype(int).to_numpy()] = True
        return ok


def make_policy(taxonomy, cfg: dict):
    """Policy from config `assignment.acceptance` ({policy, model_manifest, model_path, probation_gate}); default calibrated."""
    a = (cfg.get("assignment") or {}).get("acceptance") or {}
    kind = a.get("policy", CALIBRATED)
    if kind == CALIBRATED:
        return CalibratedAcceptance(taxonomy)
    if kind == LEARNED:
        if not a.get("model_manifest"):
            raise ValueError("assignment.acceptance.policy=learned needs assignment.acceptance.model_manifest")
        return LearnedAcceptance(taxonomy, L.LearnedAcceptanceModel.load(a["model_manifest"], a.get("model_path")), a.get("probation_gate"))
    raise ValueError(f"unknown acceptance policy {kind!r} (use {CALIBRATED!r} or {LEARNED!r})")
