import numpy as np
import scipy.sparse as sp

from taxonomy_classifier.assignment.classifier import FrozenAssigner
from taxonomy_classifier.assignment.sparse import BM25


def toy_assigner(n_reject=1):
    Dp = np.eye(4, dtype=np.float32)                   # rows: CAT_A, CAT_B, CAT_C, reject
    Sp = sp.csr_matrix(np.eye(4, dtype=np.float32))
    return FrozenAssigner(["CAT_A", "CAT_B", "CAT_C"], {"CAT_A": None, "CAT_B": None, "CAT_C": None}, Dp[: 3 + n_reject],
                          Sp[: 3 + n_reject], n_reject, [0.6, 0.6, 0.6], 0.0, 0.0, 5, 60)


def unit(v):
    v = np.asarray(v, np.float32); return v / np.linalg.norm(v)


def test_assign_statuses():
    a = toy_assigner()
    X = np.stack([unit([1, 0.9, 0, 0]), unit([0, 0, 0, 1]), unit([0.3, 0.3, 0.3, 0.2]), unit([1, 0, 0, 0])])
    B = sp.csr_matrix(X)
    res = a.assign(X, B, np.array([False, False, False, True]), ["t0", "t1", "t2", "t3"])
    assert res[0]["status"] == "ASSIGNED" and res[0]["direct"] == ["CAT_A", "CAT_B"]      # multi-label
    assert res[1]["status"] == "REJECT_NONSEMANTIC"                                        # closest to the reject prototype
    assert res[2]["status"] == "UNRESOLVED"                                                # below every dense floor
    assert res[3]["status"] == "BYPASS_ADMINISTRATIVE"


def test_parent_child_keeps_most_specific():
    a = FrozenAssigner(["P", "C"], {"P": None, "C": "P"}, np.eye(2, dtype=np.float32), sp.csr_matrix(np.eye(2, dtype=np.float32)),
                       0, [0.1, 0.1], 0.0, 0.0, 5, 60)
    X = np.stack([unit([1, 1])])
    r = a.assign(X, sp.csr_matrix(X), np.array([False]), ["t"])[0]
    assert r["direct"] == ["C"] and r["inherited"] == ["P"]


def test_bm25_frozen_state_roundtrip(cfg):
    texts = [f"deferred tax assets valuation allowance item {i}" for i in range(10)] + [f"revolving credit facility notes {i}" for i in range(10)]
    cfg = {**cfg, "assignment": {**cfg["assignment"], "bm25": {"k1": 1.2, "b": 0.75, "min_df": 2, "max_df": 0.9}}}
    bm = BM25.fit(texts, cfg)
    bm2 = BM25.from_state(bm.state())
    new = ["The valuation allowance on deferred tax assets increased", "unrelated words only"]
    assert (bm.transform(new) != bm2.transform(new)).nnz == 0
