"""Sparse (BM25) category ranking.

The BM25 vocabulary and IDF are fitted once on the taxonomy-build corpus and frozen with the taxonomy
(bm25_vocabulary.json), so new chunks are scored against exactly the vocabulary the category profiles were built with.
Category sparse profiles = mean BM25 vector of the members, truncated to the top `profile_terms` terms, L2-normalised.
"""

from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from .dense import l2rows

TOKEN_PATTERN = r"(?u)\b[a-z][a-z\-]{1,}\b"


class BM25:
    def __init__(self, k1: float, b: float, cv: CountVectorizer, idf: np.ndarray, avgdl: float):
        self.k1, self.b, self.cv, self.idf, self.avgdl = k1, b, cv, idf.astype(np.float32), float(avgdl)
        self.vocab = np.array(cv.get_feature_names_out())

    @classmethod
    def fit(cls, texts_train, cfg) -> "BM25":
        b = cfg["assignment"]["bm25"]
        cv = CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1, 2),
                             min_df=b["min_df"], max_df=b["max_df"], dtype=np.float32)
        TF = cv.fit_transform([cls._clean(t) for t in texts_train])
        n = TF.shape[0]; df = np.asarray((TF > 0).sum(0)).ravel()
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        return cls(b["k1"], b["b"], cv, idf, float(np.asarray(TF.sum(1)).ravel().mean()))

    def state(self) -> dict:
        return {"k1": self.k1, "b": self.b, "avgdl": self.avgdl, "token_pattern": TOKEN_PATTERN, "ngram_range": [1, 2],
                "stop_words": "sklearn ENGLISH_STOP_WORDS", "vocabulary": self.vocab.tolist(), "idf": [float(v) for v in self.idf]}

    @classmethod
    def from_state(cls, st: dict) -> "BM25":
        cv = CountVectorizer(token_pattern=st["token_pattern"], stop_words=list(ENGLISH_STOP_WORDS), ngram_range=tuple(st["ngram_range"]),
                             vocabulary=st["vocabulary"], dtype=np.float32)
        return cls(st["k1"], st["b"], cv, np.array(st["idf"], np.float32), st["avgdl"])

    @staticmethod
    def _clean(t):
        return re.sub(r"\s+", " ", str(t)).lower()

    def transform(self, texts):
        TF = self.cv.transform([self._clean(t) for t in texts]).tocsr()
        dl = np.asarray(TF.sum(1)).ravel()
        B = TF.copy(); rl = np.repeat(dl / self.avgdl, np.diff(B.indptr))
        B.data = B.data * (self.k1 + 1) / (B.data + self.k1 * (1 - self.b + self.b * rl))
        return l2rows(B @ sp.diags(self.idf)).tocsr().astype(np.float32)


def sparse_profiles(B, groups, mask=None, terms=300):
    rows = []
    for m in groups:
        m = m if mask is None else m[mask[m]]
        v = np.asarray(B[m].mean(0)).ravel() if len(m) else np.zeros(B.shape[1])
        if (v > 0).sum() > terms:
            v[v < np.partition(v, -terms)[-terms]] = 0
        rows.append(sp.csr_matrix(v))
    return l2rows(sp.vstack(rows).tocsr())
