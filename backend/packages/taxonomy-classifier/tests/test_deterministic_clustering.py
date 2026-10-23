"""UMAP of rows that repeat (a footer in many documents) is the same on every call."""

import hashlib

import numpy as np
import pytest

pytest.importorskip("umap")
from taxonomy_classifier.discovery.clustering import umap_embed  # noqa: E402

PARAMS = {"n_neighbors": 5, "n_components": 2, "min_dist": 0.0, "metric": "cosine"}


def _distinct(n=30, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 16)).astype(np.float32)
    return base / np.linalg.norm(base, axis=1, keepdims=True)


def _with_footer(copies=12):
    """30 distinct rows plus one row repeated ``copies`` times (at least n_neighbors: its own piece of the graph)."""
    base = _distinct()
    footer = _distinct(1, seed=99)
    return np.vstack([base, np.repeat(footer, copies, axis=0)])


def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_a_large_group_of_repeats_gives_the_same_embedding_every_time():
    X = _with_footer()
    assert len({_hash(umap_embed(X, 42, **PARAMS)) for _ in range(4)}) == 1


def test_a_large_group_of_repeats_stays_together():
    """Each repeat stays its own point, close to the others (the group keeps its weight and can form its own cluster).

    That the group is not merged into a real topic is checked on structured data by
    test_airweave_integration.py::test_build_produces_taxonomy_and_artifacts (3 topics plus the signature cluster).
    """
    X = _with_footer()
    e = umap_embed(X, 42, **PARAMS)
    group, rest = e[30:], e[:30]
    spread = np.linalg.norm(group - group.mean(0), axis=1).max()
    rest_spread = np.linalg.norm(rest - rest.mean(0), axis=1).max()
    assert spread < 0.25 * rest_spread


def test_small_groups_of_repeats_are_unchanged_from_plain_umap():
    import umap

    base = _distinct()
    X = np.vstack([base, base[:6], base[:6]])   # each repeated row appears 3 times, fewer than n_neighbors
    plain = umap.UMAP(random_state=7, **PARAMS).fit_transform(X)
    assert np.allclose(umap_embed(X, 7, **PARAMS), plain)


def test_rows_without_repeats_are_unchanged_from_plain_umap():
    import umap

    X = _distinct()
    plain = umap.UMAP(random_state=7, **PARAMS).fit_transform(X)
    assert np.allclose(umap_embed(X, 7, **PARAMS), plain)
