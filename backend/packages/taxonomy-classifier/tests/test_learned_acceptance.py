"""Learned acceptance: feature schema/order, model manifest loading and failure modes, threshold, zero-label, probation gate.

Neutral data only: the taxonomy is the recipes / software / travel build from test_generic_corpus; the "model" is a
constant-probability classifier so thresholds can be tested exactly. No API calls.
"""

import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from taxonomy_classifier.assignment import learned as L
from taxonomy_classifier.assignment.acceptance import CalibratedAcceptance, LearnedAcceptance, make_policy
from taxonomy_classifier.taxonomy import artifacts, maturity

from .test_generic_corpus import generic  # noqa: F401  (module-scoped neutral build)

pytestmark = pytest.mark.slow


def write_model(tmp_path, p, features=L.FEATURE_NAMES, schema=L.FEATURE_SCHEMA_VERSION, threshold=0.80, names=None):
    y = np.array([1] * int(round(p * 100)) + [0] * (100 - int(round(p * 100))))
    m = DummyClassifier(strategy="prior").fit(pd.DataFrame(np.zeros((100, len(features))), columns=list(features)), y)
    f = tmp_path / f"model_{p}.pkl"
    pickle.dump({"model": m, "features": list(features)}, open(f, "wb"))
    man = {"model_id": f"const_{p}", "feature_schema_version": schema, "feature_names": list(names or L.FEATURE_NAMES), "model_type": "dummy",
           "model_file": f.name, "model_sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "threshold": threshold, "max_labels_per_child": 1}
    mp = tmp_path / f"manifest_{p}.json"; json.dump(man, open(mp, "w"))
    return mp


def test_feature_schema_is_fixed_and_ordered():
    assert len(L.FEATURE_NAMES) == 40 and len(set(L.FEATURE_NAMES)) == 40
    assert L.FEATURE_NAMES[:3] == ("dense", "dense_single", "dense_rank") and L.FEATURE_NAMES[-1] == "form_LIST"
    pf = pd.DataFrame([{**{f: 0.0 for f in L.FEATURE_NAMES if not f.startswith(("scope_", "form_"))}, "cat_scope": "GLOBAL", "content_form": "TABLE"}])
    d = L.design(pf)
    assert list(d.columns) == list(L.FEATURE_NAMES)
    assert d.scope_GLOBAL.iat[0] == 1.0 and d.form_TABLE.iat[0] == 1.0 and d.form_NARRATIVE.iat[0] == 0.0


def test_model_loads_and_mismatches_fail_loudly(tmp_path):
    ok = L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.9))
    assert ok.threshold == 0.80 and ok.max_labels == 1
    with pytest.raises(L.FeatureSchemaMismatch):
        L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.91, schema="learned-acceptance-features/0"))
    with pytest.raises(L.FeatureSchemaMismatch):                                   # manifest lists a different order
        L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.92, names=list(reversed(L.FEATURE_NAMES))))
    with pytest.raises(L.FeatureSchemaMismatch):                                   # pickled model trained on another order
        L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.93, features=tuple(reversed(L.FEATURE_NAMES))))
    mp = write_model(tmp_path, 0.94); man = json.load(open(mp)); man["model_sha256"] = "0" * 64; json.dump(man, open(mp, "w"))
    with pytest.raises(ValueError, match="sha256"):
        L.LearnedAcceptanceModel.load(mp)
    with pytest.raises(L.FeatureSchemaMismatch):
        ok.predict(pd.DataFrame(np.zeros((1, 40)), columns=list(reversed(L.FEATURE_NAMES))))


def _child_inputs(generic, texts):  # noqa: F811
    from tests.test_generic_corpus import bow_vector
    tax = generic["store"].load("my_custom_corpus", "v1")
    X = np.stack([bow_vector(t) for t in texts]); X /= np.linalg.norm(X, axis=1, keepdims=True)
    return tax, X.astype(np.float32), tax.bm25.transform(texts)


TEXTS = ["First whisk the eggs, then whisk the eggs again. First preheat the oven, then preheat the oven again.",
         "First install the package, then install the package again. First run the command line, then run the command line again."]


def test_threshold_and_zero_label(generic, tmp_path):  # noqa: F811
    tax, X, B = _child_inputs(generic, TEXTS)
    admin = np.array([False, False])
    hi = LearnedAcceptance(tax, L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.85)))
    out = hi.assign(X, B, admin, ["c0", "c1"], tokens=[40, 40], content_forms=["NARRATIVE"] * 2)
    assert [r["status"] for r in out] == ["ASSIGNED", "ASSIGNED"] and all(len(r["direct"]) == 1 for r in out)   # max 1 label
    lo = LearnedAcceptance(tax, L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.75)))
    out = lo.assign(X, B, admin, ["c0", "c1"], tokens=[40, 40], content_forms=["NARRATIVE"] * 2)
    assert [r["status"] for r in out] == ["UNRESOLVED", "UNRESOLVED"] and all(r["direct"] == [] for r in out)   # never forced
    out = hi.assign(X, B, np.array([True, False]), ["c0", "c1"], tokens=[40, 40], content_forms=["NARRATIVE"] * 2)
    assert out[0]["status"] == "BYPASS_ADMINISTRATIVE"


def test_probation_gate_rule():
    g = maturity.VALIDATED_PROBATION_GATE
    assert g == {"p_min": 0.95, "bm25_rank_max": 2, "dense_rank_max": 2}
    assert maturity.gate_ok(0.81, 9, 9, maturity.MATURE, 0.80)                        # mature: threshold only
    assert not maturity.gate_ok(0.79, 1, 1, maturity.MATURE, 0.80)
    assert not maturity.gate_ok(0.94, 1, 1, maturity.PROBATIONARY, 0.80)              # probationary: P >= 0.95
    assert not maturity.gate_ok(0.99, 3, 1, maturity.PROBATIONARY, 0.80)              # and BM25 rank <= 2
    assert not maturity.gate_ok(0.99, 1, 3, maturity.PROBATIONARY, 0.80)              # and dense rank <= 2
    assert maturity.gate_ok(0.95, 2, 2, maturity.PROBATIONARY, 0.80)
    for s in maturity.RETIRED:
        assert not maturity.gate_ok(1.0, 1, 1, s, 0.80)


def test_probationary_category_is_gated_in_the_policy(generic, tmp_path):  # noqa: F811
    tax, X, B = _child_inputs(generic, TEXTS[:1])
    pol = LearnedAcceptance(tax, L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.90)))
    out = pol.assign(X, B, np.array([False]), ["c0"], tokens=[40], content_forms=["NARRATIVE"])
    assert out[0]["status"] == "ASSIGNED"
    target = out[0]["direct"][0]
    pol.rep.statuses[pol.rep.index(target)] = maturity.PROBATIONARY                   # status is data, not a name list
    out2 = pol.assign(X, B, np.array([False]), ["c0"], tokens=[40], content_forms=["NARRATIVE"])
    assert target not in out2[0]["direct"]                                            # P 0.90 < 0.95: blocked
    pol96 = LearnedAcceptance(tax, L.LearnedAcceptanceModel.load(write_model(tmp_path, 0.96)), rep=pol.rep)
    pf = pol96.score_pairs(X, B, np.array([False]), [40], ["NARRATIVE"])
    row = pf[pf.category_id == target].iloc[0]
    assert bool(row.gate_ok) == (row.bm25_rank <= 2 and row.dense_rank <= 2)


def test_policy_selection_from_config(generic, tmp_path):  # noqa: F811
    tax = generic["store"].load("my_custom_corpus", "v1")
    assert isinstance(make_policy(tax, {"assignment": {}}), CalibratedAcceptance)
    cfg = {"assignment": {"acceptance": {"policy": "learned", "model_manifest": str(write_model(tmp_path, 0.9))}}}
    p = make_policy(tax, cfg)
    assert isinstance(p, LearnedAcceptance) and p.provenance()["acceptance_model"] == "const_0.9"
    with pytest.raises(ValueError):
        make_policy(tax, {"assignment": {"acceptance": {"policy": "learned"}}})


def test_built_artifact_carries_representation(generic):  # noqa: F811
    tax = generic["store"].load("my_custom_corpus", "v1")
    rep = artifacts.load_representation(tax)
    assert rep.ids == [c["category_id"] for c in tax.categories]
    assert all(c["status"] == maturity.MATURE and "stats" in c for c in tax.categories)
    assert all(len(p) >= 1 for p in rep.prototypes) and all(len(e) == c["stats"]["cat_members"] for e, c in zip(rep.exemplars, tax.categories))


def test_classify_output_records_provenance(generic, tmp_path):  # noqa: F811
    from taxonomy_classifier import TaxonomyEngine
    from tests.test_generic_corpus import BowEmbeddings, corpus
    cfg = {**generic["cfg"], "assignment": {**generic["cfg"]["assignment"],
                                            "acceptance": {"policy": "learned", "model_manifest": str(write_model(tmp_path, 0.9))}}}
    eng = TaxonomyEngine.load("my_custom_corpus", cfg, generic["store"], version="v1", embed_transport=BowEmbeddings())
    rec = eng.classify(corpus(n_docs=1, chunks_per_doc=1, seed=3, prefix="P")).records[0]
    assert rec["assignment"] == {"acceptance_policy": "learned", "acceptance_model": "const_0.9", "threshold": 0.8}
    assert all(t["category_status"] == "MATURE" and t["sources"] == ["LEARNED_ACCEPTANCE"] for t in rec["topics"])
