"""Multi-label assignment: dense ranking + BM25 ranking -> RRF -> calibrated acceptance. NO LLM.

Build time (HybridAssigner.calibrate), leave-group-out over the build corpus:
  dense floor   = lowest cosine at which held-out ADMINISTRATIVE-tagged non-members are top-1 assigned <= junk_max,
                  raised per category to its P10 held-out member cosine (>= category_min_heldout members, >= 3 groups)
  RRF tau       = lowest value with held-out multi-label hierarchy-aware precision >= precision_min and junk <= junk_max
  sibling delta = P25 of held-out top1-top2 RRF gaps (parent fallback)
Run time (FrozenAssigner.assign): a category is accepted for a taxonomy chunk when it is in the RRF top-`window`,
outranks the best reject (structural/administrative) prototype, has dense cosine >= its floor and RRF >= tau.

FrozenAssigner holds only the frozen state (prototypes, profiles, thresholds); it is saved/loaded by taxonomy/artifacts.py.
"""

from __future__ import annotations

import warnings

import numpy as np

from .dense import dense_protos
from .rrf import ranks, rrf
from .sparse import sparse_profiles


class CalibrationError(RuntimeError):
    pass


class CalibrationWarning(UserWarning):
    pass


def ancestors(parent: dict) -> dict:
    anc = {}
    for c in parent:
        s, p = set(), parent.get(c)
        while p:
            s.add(p); p = parent.get(p)
        anc[c] = s
    return anc


class FrozenAssigner:
    def __init__(self, pos_ids, parent, Dp, Sp, n_reject, floors, tau, delta, window, rrf_k):
        """Dp / Sp: dense / sparse prototypes, rows = positive categories (pos_ids order) followed by n_reject reject prototypes."""
        self.pos_ids, self.parent = list(pos_ids), dict(parent)
        self.anc = ancestors(self.parent)
        self.Dp, self.Sp = Dp, Sp
        self.P, self.n_reject = len(self.pos_ids), int(n_reject)
        self.fl, self.tau, self.delta = np.asarray(floors, np.float32), float(tau), float(delta)
        self.window, self.rrf_k = int(window), int(rrf_k)

    def scores(self, X, B):
        Sd = X @ self.Dp.T; Ss = (B @ self.Sp.T).toarray().astype(np.float32)
        return Sd, Ss, rrf(Sd, Ss, self.rrf_k)

    def _cand(self, S, Sd):
        P = self.P; Sp = S[:, :P]
        rj = S[:, P:].max(1) if self.n_reject else np.full(len(S), -1.0)
        rejected = rj >= Sp.max(1)
        ok = (ranks(Sp) <= self.window) & (Sp > rj[:, None]) & (Sd[:, :P] >= self.fl[None, :]) & ~rejected[:, None]
        return ok, rejected

    def assign(self, X, B, admin, ids) -> list[dict]:
        if len(X) == 0:
            return []
        Sd, Ss, S = self.scores(X, B)
        ok, rejected = self._cand(S, Sd)
        P = self.P; Sp = S[:, :P]
        acc = ok & (Sp >= self.tau - 1e-6)
        od, os_, orr = np.argsort(-Sd[:, :P], 1), np.argsort(-Ss[:, :P], 1), np.argsort(-Sp, 1)
        out = []
        for i in range(len(X)):
            rec = {"taxonomy_chunk_id": ids[i], "dense_top": [(self.pos_ids[j], round(float(Sd[i, j]), 4)) for j in od[i, :5]],
                   "sparse_top": [(self.pos_ids[j], round(float(Ss[i, j]), 4)) for j in os_[i, :5]],
                   "rrf_top": [(self.pos_ids[j], round(float(Sp[i, j]), 6)) for j in orr[i, :5]]}
            if admin[i]:
                out.append({**rec, "status": "BYPASS_ADMINISTRATIVE", "direct": [], "inherited": [], "scores": [], "parent_fallback": False}); continue
            if rejected[i]:
                out.append({**rec, "status": "REJECT_NONSEMANTIC", "direct": [], "inherited": [], "scores": [], "parent_fallback": False}); continue
            accj = [j for j in orr[i, :self.window] if acc[i, j]]
            if not accj:
                out.append({**rec, "status": "UNRESOLVED", "direct": [], "inherited": [], "scores": [], "parent_fallback": False}); continue
            ks = [self.pos_ids[j] for j in accj]; sc = {self.pos_ids[j]: float(Sp[i, j]) for j in accj}
            fb = False
            p0 = self.parent.get(ks[0])
            if len(ks) >= 2 and p0 and p0 == self.parent.get(ks[1]) and sc[ks[0]] - sc[ks[1]] < self.delta:
                ks = [p0] + [k for k in ks if self.parent.get(k) != p0 and k != p0]; sc[p0] = sc.get(p0, max(sc.values())); fb = True
            direct = list(dict.fromkeys(k for k in ks if not any(k in self.anc.get(o, set()) for o in ks if o != k)))
            inh = sorted({x for k in direct for x in self.anc.get(k, set())} - set(direct))
            out.append({**rec, "status": "ASSIGNED", "direct": direct, "inherited": inh, "scores": [round(sc.get(k, np.nan), 6) for k in direct],
                        "parent_fallback": fb})
        return out


class HybridAssigner(FrozenAssigner):
    """Build-time assigner: builds prototypes from the build corpus and calibrates thresholds."""

    def __init__(self, cfg, X_train, B_train, group_train, admin_train, cat_ids, cat_groups, parent, reject_groups):
        """cat_groups / reject_groups: index arrays into the TRAIN matrices. parent: cat_id -> parent cat_id or None."""
        a = cfg["assignment"]; self.cfg = a
        keep = [i for i, g in enumerate(cat_groups) if len(g) >= a["calibration"]["min_members_for_prototype"]]
        self.groups = [cat_groups[i] for i in keep]
        self.rej = [g for g in reject_groups if len(g) >= 10]
        self.allg = self.groups + self.rej
        self.Xt, self.Bt, self.comp, self.admin = X_train, B_train, group_train, admin_train
        Dp = dense_protos(X_train, self.allg); Sp = sparse_profiles(B_train, self.allg, terms=a["profile_terms"])
        full_parent = {c: parent.get(c) for c in cat_ids}
        super().__init__([cat_ids[i] for i in keep], full_parent, Dp, Sp, len(self.rej), np.zeros(len(keep)), 0.0, 0.0, a["window"], a["rrf_k"])

    def calibrate(self, seed):
        a = self.cfg; cal = a["calibration"]; P = self.P
        n = len(self.Xt); comps = np.array(sorted(set(self.comp)))
        rng = np.random.default_rng(seed); sh = comps.copy(); rng.shuffle(sh)
        Sd = np.zeros((n, len(self.allg)), np.float32); Ss = np.zeros_like(Sd)
        for f in np.array_split(sh, cal["folds"]):
            te = np.isin(self.comp, f); mask = ~te
            Sd[te] = self.Xt[te] @ dense_protos(self.Xt, self.allg, mask).T
            Ss[te] = (self.Bt[te] @ sparse_profiles(self.Bt, self.allg, mask, a["profile_terms"]).T).toarray()
        S = rrf(Sd, Ss, a["rrf_k"])
        known = np.zeros((n, P), bool)
        for j, g in enumerate(self.groups):
            known[g, j] = True
        rel = np.array([[x == y or y in self.anc[x] or x in self.anc[y] for y in self.pos_ids] for x in self.pos_ids])
        kh = (known.astype(np.int32) @ rel.astype(np.int32)) > 0
        rows = np.where(known.any(1))[0]
        junk = self.admin & ~known.any(1)
        for g in self.rej:
            junk[g] = True
        ts = Sd[:, :P].max(1); rj = Sd[:, P:].max(1) if self.rej else np.full(n, -1.0)
        status, reason = "OK", None
        if not junk.any():
            # no administrative / structural negatives outside categories: the junk-rate constraints cannot be evaluated,
            # so the search below falls back to its strictest values (dense floor 0.95, tau = max). Never silent.
            status, reason = "STRICT_FALLBACK", "empty junk pool: no administrative or reject-prototype children outside categories"
            policy = cal.get("on_empty_junk_pool", "warn_and_use_default")
            if policy == "error":
                raise CalibrationError(reason + " (assignment.calibration.on_empty_junk_pool=error)")
            if policy == "warn_and_use_default":
                warnings.warn(f"calibration: {reason}; using the strict default thresholds", CalibrationWarning, stacklevel=2)
            elif policy != "explicit_strict_fallback":
                raise ValueError(f"unknown assignment.calibration.on_empty_junk_pool {policy!r}")
        floor = 0.95
        for th in np.arange(0.55, 0.95, 0.0025):
            if ((ts >= th) & (ts > rj))[junk].mean() <= cal["junk_max"]:
                floor = float(th); break
        fl = np.full(P, floor, np.float32)
        for j in range(P):
            m = known[:, j]
            if m.sum() >= cal["category_min_heldout"] and len(set(self.comp[m])) >= 3:
                fl[j] = max(np.quantile(Sd[m, j], cal["category_quantile"]), floor)
        self.fl = fl
        ok, _ = self._cand(S, Sd)
        Spc = S[:, :P]
        vals = np.unique(np.round(Spc[ok], 6))
        grid = vals if len(vals) <= 400 else np.quantile(vals, np.linspace(0, 1, 401))
        tau = None
        for t in grid:
            acc = ok & (Spc >= t - 1e-6)
            am = acc[rows]
            prec = kh[rows][am].mean() if am.any() else np.nan
            if prec >= cal["precision_min"] and acc[junk].any(1).mean() <= cal["junk_max"]:
                tau = float(t); break
        self.tau = float(grid[-1]) if tau is None else tau
        g = np.sort(Spc, 1)
        self.delta = float(np.quantile((g[:, -1] - g[:, -2])[rows], 0.25))
        self.calib = {"dense_floor": floor, "categories_with_raised_floor": int((fl > floor).sum()), "tau_rrf": self.tau,
                      "sibling_delta": self.delta, "positive_prototypes": P, "reject_prototypes": len(self.rej),
                      "heldout_members": int(len(rows)), "junk_pool": int(junk.sum()), "status": status, "fallback_reason": reason}
        return self.calib

    def freeze(self) -> FrozenAssigner:
        return FrozenAssigner(self.pos_ids, self.parent, self.Dp, self.Sp, self.n_reject, self.fl, self.tau, self.delta, self.window, self.rrf_k)
