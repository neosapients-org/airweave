"""Reciprocal Rank Fusion of a dense and a sparse ranking: RRF(c) = 1/(k + rank_dense(c)) + 1/(k + rank_sparse(c))."""

from __future__ import annotations

import numpy as np


def ranks(S: np.ndarray) -> np.ndarray:
    """1-based rank of every column within each row (highest score = rank 1)."""
    o = np.argsort(-S, 1); R = np.empty_like(o)
    R[np.arange(len(S))[:, None], o] = np.arange(1, S.shape[1] + 1)
    return R


def rrf(Sd: np.ndarray, Ss: np.ndarray, k: int) -> np.ndarray:
    return (1.0 / (k + ranks(Sd)) + 1.0 / (k + ranks(Ss))).astype(np.float32)
