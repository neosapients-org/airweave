"""Dense category ranking: cosine between L2-normalised chunk embeddings and category prototypes (member centroids)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def l2rows(M):
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) if sp.issparse(M) else np.linalg.norm(M, axis=1)
    n[n == 0] = 1
    return sp.diags(1 / n) @ M if sp.issparse(M) else M / n[:, None]


def dense_protos(X: np.ndarray, groups, mask=None) -> np.ndarray:
    P = np.zeros((len(groups), X.shape[1]), np.float32)
    for g, m in enumerate(groups):
        m = m if mask is None else m[mask[m]]
        if len(m):
            p = X[m].mean(0); P[g] = p / np.linalg.norm(p)
    return P
