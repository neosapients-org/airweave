"""Calibration never falls back silently: an empty junk pool is recorded, warned about, or refused, per config."""

import numpy as np
import pytest
import scipy.sparse as sp

from taxonomy_classifier.assignment.classifier import CalibrationError, CalibrationWarning, HybridAssigner


def make(policy, admin_rows=0):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2, 16))
    X = np.vstack([centers[i % 2] + 0.1 * rng.normal(size=16) for i in range(60)]); X /= np.linalg.norm(X, axis=1, keepdims=True)
    B = sp.csr_matrix(np.abs(rng.normal(size=(60, 30))))
    groups_of = np.array([f"g{i % 5}" for i in range(60)], dtype=object)
    admin = np.zeros(60, bool); admin[:admin_rows] = True
    members = [np.array([i for i in range(admin_rows, 60) if i % 2 == k]) for k in (0, 1)]
    cfg = {"assignment": {"window": 5, "rrf_k": 60, "profile_terms": 30, "calibration": {
        "folds": 5, "precision_min": 0.9, "junk_max": 0.1, "category_quantile": 0.1, "category_min_heldout": 20,
        "min_members_for_prototype": 10, **({"on_empty_junk_pool": policy} if policy else {})}}}
    return HybridAssigner(cfg, X.astype(np.float32), B, groups_of, admin, ["A", "B"], members, {"A": None, "B": None}, [])


def test_default_warns_and_records_strict_fallback():
    with pytest.warns(CalibrationWarning, match="empty junk pool"):
        cal = make(None).calibrate(0)
    assert cal["status"] == "STRICT_FALLBACK" and cal["junk_pool"] == 0 and cal["dense_floor"] == 0.95 and cal["fallback_reason"]


def test_error_policy_refuses():
    with pytest.raises(CalibrationError):
        make("error").calibrate(0)


def test_explicit_fallback_is_recorded_without_warning(recwarn):
    cal = make("explicit_strict_fallback").calibrate(0)
    assert cal["status"] == "STRICT_FALLBACK" and not [w for w in recwarn if issubclass(w.category, CalibrationWarning)]


def test_normal_calibration_status_ok():
    cal = make(None, admin_rows=6).calibrate(0)
    assert cal["status"] == "OK" and cal["junk_pool"] > 0 and cal["fallback_reason"] is None
