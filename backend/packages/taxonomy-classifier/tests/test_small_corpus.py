"""Calibration on a corpus that is too small: a clear error or a degraded success, never a crash.

Synthetic data, no API calls. The three cases that crashed on the Initech test (18 documents):
no category big enough to calibrate, a single category, and no administrative pieces to set the floor.
"""
import copy
import pathlib
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
import yaml

from taxonomy_classifier.assignment.classifier import CorpusTooSmall, HybridAssigner

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "config" / "taxonomy.example.yaml"


def corpus(sizes, junk=0, d=16, vocab=40, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n = sum(sizes) + junk
    X, B = np.zeros((n, d), np.float32), np.zeros((n, vocab), np.float32)
    i = 0
    for s in sizes:
        centre, words = rng.normal(size=d), rng.integers(0, vocab, 6)
        for _ in range(s):
            X[i] = centre + noise * rng.normal(size=d)
            B[i, words] = 1 + rng.random(6)
            i += 1
    admin = np.zeros(n, bool)
    for _ in range(junk):
        X[i] = rng.normal(size=d)
        B[i, rng.integers(0, vocab, 4)] = 1
        admin[i] = True
        i += 1
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, sp.csr_matrix(B), np.array([f"doc{j}" for j in range(n)], object), admin


def calibrate(sizes, junk=0, noise=0.15, min_members=4):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["assignment"]["calibration"].update(folds=3, min_members_for_prototype=min_members, category_min_heldout=3)
    X, B, groups_of, admin = corpus(sizes, junk, noise=noise)
    groups, start = [], 0
    for s in sizes:
        groups.append(np.arange(start, start + s))
        start += s
    ids = [f"C{k}" for k in range(len(sizes))]
    assigner = HybridAssigner(copy.deepcopy(cfg), X, B, groups_of, admin, ids, groups, {k: None for k in ids}, [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return assigner.calibrate(42)


def test_no_category_big_enough_says_so():
    with pytest.raises(CorpusTooSmall, match=r"no category has the 4 members .*largest has 3"):
        calibrate([3, 3, 3])


def test_a_single_category_calibrates_with_no_sibling_gap():
    calib = calibrate([9, 3, 3])
    assert calib["positive_prototypes"] == 1 and calib["sibling_delta"] == 0.0


def test_no_administrative_pieces_and_no_candidate_says_so():
    with pytest.raises(CorpusTooSmall, match=r"no chunk passes the dense floor \(0\.95\).*administrative"):
        calibrate([12, 12], noise=0.7)


def test_the_same_corpus_calibrates_once_it_has_administrative_pieces():
    calib = calibrate([12, 12], junk=20, noise=0.7)
    assert calib["status"] == "OK" and calib["dense_floor"] < 0.95


def test_an_ordinary_corpus_is_unchanged():
    calib = calibrate([12, 12], junk=20)
    assert calib["status"] == "OK" and calib["positive_prototypes"] == 2
