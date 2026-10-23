"""Learned acceptance: candidate pairs (taxonomy child, category) -> 40 features -> frozen classifier -> P(correct).

The mechanism is generic: every feature is computed from the taxonomy artifact (category prototypes, exemplars, sparse
profiles, statistics, scope, family, sink flag) and the child (embedding, BM25 vector, token count, content form).
A trained model is data: it is loaded from a manifest that pins its feature schema, feature order, threshold and file hash.

Candidates per child = dense top-5 ∪ BM25 top-5 ∪ RRF top-5 over all categories (dense = multi-prototype max cosine);
administrative children are never scored. This is a line-for-line port of the validated research scorer
(provenance and exact definition: docs/VALIDATED_BASELINE_SPEC.md).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .rrf import ranks, rrf

FEATURE_SCHEMA_VERSION = "learned-acceptance-features/1"
FEATURE_NAMES = (
    "dense", "dense_single", "dense_rank", "exemplar_max", "exemplar_top5",
    "bm25", "bm25_rank", "bm25_norm",
    "rrf", "rrf_rank",
    "gap_top1_top2", "gap_to_next", "rel_dense_to_top", "rel_bm25_to_top", "rel_rrf_to_top", "agree_top1_dense_bm25",
    "in_bm25_top3", "in_dense_top3", "top1_any", "top1_all", "same_family_as_dense_top1", "n_candidates",
    "reject_sim", "margin_over_reject", "reject_outranks_all",
    "cat_members", "cat_companies", "cat_documents", "cat_dominant_share", "cat_dispersion", "cat_is_child", "cat_is_sink",
    "scope_GLOBAL", "scope_INDUSTRY",
    "tokens", "multi_topic_proxy", "form_NARRATIVE", "form_TABLE", "form_MIXED_TABLE_AND_NARRATIVE", "form_LIST",
)
STAT_KEYS = ("cat_members", "cat_companies", "cat_documents", "cat_dominant_share", "cat_dispersion", "cat_is_child", "cat_is_sink")
FORMS = ("NARRATIVE", "TABLE", "MIXED_TABLE_AND_NARRATIVE", "LIST", "OTHER")
TOP_K = 5          # candidate union: dense / BM25 / RRF top-5
RRF_K = 60


class FeatureSchemaMismatch(ValueError):
    pass


def score_matrices(rep, X: np.ndarray, B) -> dict:
    """rep: taxonomy.representation.CategoryRepresentation. Returns (n, P) matrices + reject similarity (n,)."""
    n, P = len(X), len(rep.ids)
    multi = np.zeros((n, P), np.float32); exm = np.zeros((n, P), np.float32); ex5 = np.zeros((n, P), np.float32)
    single = (X @ rep.centroids.T).astype(np.float32) if P else np.zeros((n, 0), np.float32)
    for j in range(P):
        multi[:, j] = (X @ rep.prototypes[j].T).max(1)
        E = rep.exemplars[j]
        if not len(E):
            exm[:, j] = np.nan; ex5[:, j] = np.nan; continue
        for s in range(0, n, 4096):
            S = X[s:s + 4096] @ E.T
            exm[s:s + 4096, j] = S.max(1); k = min(5, S.shape[1]); ex5[s:s + 4096, j] = np.partition(S, -k, 1)[:, -k:].mean(1)
    sparse = (B @ rep.sparse.T).toarray().astype(np.float32) if P else np.zeros((n, 0), np.float32)
    rej = (X @ rep.reject.T).max(1) if len(rep.reject) else np.zeros(n, np.float32)
    return {"multi": multi, "single": single, "exmax": exm, "ex5": ex5, "sparse": sparse, "rej": rej}


def pair_features(rep, M: dict, tokens, content_forms, admin) -> pd.DataFrame:
    """One row per (child k, candidate category). `k` indexes the input children."""
    ids, P = rep.ids, len(rep.ids)
    cols = ["k", "category_id", "cat_scope", "content_form", *[f for f in FEATURE_NAMES if not f.startswith(("scope_", "form_"))]]
    if P == 0:
        return pd.DataFrame(columns=cols)
    sd, ss = M["multi"], M["sparse"]; S = rrf(sd, ss, RRF_K)
    sp_full = rrf(np.concatenate([sd, M["rej"][:, None]], 1), np.concatenate([ss, np.zeros((len(ss), 1), np.float32)], 1), RRF_K)
    Rd, Rs, Rr = ranks(sd), ranks(ss), ranks(S)
    sds = np.sort(sd, 1); ssm = ss.max(1); Sm = S.max(1)
    rows = []
    for k in range(len(sd)):
        if admin[k]:
            continue
        cand = sorted(set(np.where(Rd[k] <= TOP_K)[0]) | set(np.where(Rs[k] <= TOP_K)[0]) | set(np.where(Rr[k] <= TOP_K)[0]))
        d1, s1, r1 = int(np.argmax(sd[k])), int(np.argmax(ss[k])), int(np.argmax(S[k]))
        g12 = sds[k, -1] - sds[k, -2] if P > 1 else 0.0
        cf = content_forms[k]
        for j in cand:
            lower = sd[k][sd[k] < sd[k, j]]
            fam = rep.families[j]
            rows.append({"k": k, "category_id": ids[j], "dense": sd[k, j], "dense_single": M["single"][k, j], "dense_rank": int(Rd[k, j]),
                         "exemplar_max": M["exmax"][k, j], "exemplar_top5": M["ex5"][k, j], "gap_top1_top2": g12,
                         "gap_to_next": sd[k, j] - lower.max() if len(lower) else 0.5, "rel_dense_to_top": sd[k, j] - sds[k, -1],
                         "bm25": ss[k, j], "bm25_rank": int(Rs[k, j]), "bm25_norm": ss[k, j] / ssm[k] if ssm[k] > 0 else 0.0,
                         "rel_bm25_to_top": ss[k, j] - ssm[k], "rrf": S[k, j], "rrf_rank": int(Rr[k, j]), "rel_rrf_to_top": S[k, j] - Sm[k],
                         "reject_sim": M["rej"][k], "margin_over_reject": sd[k, j] - M["rej"][k],
                         "reject_outranks_all": bool(sp_full[k, -1] >= sp_full[k, :P].max()), **rep.stats[j], "cat_scope": rep.scopes[j],
                         "tokens": int(tokens[k]), "content_form": cf if cf in FORMS else "OTHER", "multi_topic_proxy": float(g12 < 0.01),
                         "n_candidates": len(cand), "agree_top1_dense_bm25": d1 == s1, "in_bm25_top3": Rs[k, j] <= 3, "in_dense_top3": Rd[k, j] <= 3,
                         "top1_any": j in (d1, s1, r1), "top1_all": j == d1 == s1 == r1,
                         "same_family_as_dense_top1": (fam == rep.families[d1] and j != d1) if fam else False})
    return pd.DataFrame(rows, columns=None if rows else cols)


def design(pf: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix in FEATURE_NAMES order (one-hot scope / content form; missing exemplar scores -> -1)."""
    d = pf.copy()
    for s in ("GLOBAL", "INDUSTRY"):
        d[f"scope_{s}"] = (d.cat_scope == s).astype(float)
    for f in ("NARRATIVE", "TABLE", "MIXED_TABLE_AND_NARRATIVE", "LIST"):
        d[f"form_{f}"] = (d.content_form == f).astype(float)
    return d[list(FEATURE_NAMES)].astype(float).fillna(-1.0)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _check_decision_settings(mp: Path, man: dict) -> None:
    """The manifest's threshold is a probability and its label limit a positive count; anything else would silently turn
    acceptance into all-or-none or no labels, so the model is refused before it is loaded."""
    try:
        threshold = float(man["threshold"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{mp}: threshold is missing or not a number") from None
    if not 0.0 <= threshold <= 1.0:                     # also false for NaN
        raise ValueError(f"{mp}: threshold {man['threshold']!r} is not a probability in [0, 1]")
    limit = man.get("max_labels_per_child", 1)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"{mp}: max_labels_per_child {limit!r} is not a whole number of at least 1")


@dataclass
class LearnedAcceptanceModel:
    """A frozen acceptance classifier plus the manifest that pins how it may be used.

    manifest.json: {"model_id", "feature_schema_version", "feature_names", "model_type", "model_file", "model_sha256",
                    "threshold", "max_labels_per_child", "training": {...}, "intended_use": "..."}
    The model file is a pickle {"model": <sklearn classifier>, "features": [...]}: it is loaded only after its sha256
    matches the manifest (pickles execute code on load; never load an unverified file).
    """
    model_id: str
    manifest: dict
    model: object = field(repr=False)
    path: Path | None = None

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"])

    @property
    def max_labels(self) -> int:
        return int(self.manifest.get("max_labels_per_child", 1))

    @classmethod
    def load(cls, manifest_path: str | Path, model_path: str | Path | None = None) -> "LearnedAcceptanceModel":
        mp = Path(manifest_path)
        man = json.load(open(mp))
        if man.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise FeatureSchemaMismatch(f"{mp}: feature_schema_version {man.get('feature_schema_version')!r}, engine computes {FEATURE_SCHEMA_VERSION!r}")
        if tuple(man.get("feature_names", ())) != FEATURE_NAMES:
            raise FeatureSchemaMismatch(f"{mp}: feature_names differ from the engine's feature order")
        _check_decision_settings(mp, man)
        f = Path(model_path) if model_path else mp.parent / man["model_file"]
        if not f.exists():
            raise FileNotFoundError(f"acceptance model file {f} not found (manifest {mp}); supply it via config assignment.acceptance.model_path")
        if _sha(f) != man["model_sha256"]:
            raise ValueError(f"acceptance model {f}: sha256 does not match the manifest; refusing to load")
        obj = pickle.load(open(f, "rb"))
        model, feats = (obj["model"], obj.get("features")) if isinstance(obj, dict) else (obj, None)
        if feats is not None and tuple(feats) != FEATURE_NAMES:
            raise FeatureSchemaMismatch(f"{f}: the pickled model was trained on a different feature order")
        if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
            raise FeatureSchemaMismatch(f"{f}: model expects {model.n_features_in_} features, schema has {len(FEATURE_NAMES)}")
        return cls(man["model_id"], man, model, f)

    def predict(self, Xd: pd.DataFrame) -> np.ndarray:
        if list(Xd.columns) != list(FEATURE_NAMES):
            raise FeatureSchemaMismatch("design matrix columns are not in FEATURE_NAMES order")
        if not len(Xd):
            return np.zeros(0)
        return self.model.predict_proba(Xd)[:, 1]
