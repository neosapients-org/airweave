"""Deterministic offline stand-ins for the embedding and chat APIs + a synthetic Airweave corpus.

Used by the test-suite and by `scripts/run_end_to_end_demo.py --offline` to exercise the full pipeline with no API key
and no spend. The fake reviewer is NOT a taxonomy method: it exists only to drive the code paths.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

DIM = 48

TOPICS = {
    "tax": ["deferred tax assets", "valuation allowance", "effective tax rate", "income tax expense", "tax credit carryforwards",
            "uncertain tax positions", "statutory rate", "taxable income"],
    "fair": ["fair value hierarchy", "level 3 inputs", "unobservable inputs", "quoted prices", "valuation techniques",
             "discounted cash flow", "market comparables", "measured at fair value"],
    "debt": ["revolving credit facility", "senior notes", "debt maturities", "interest rate", "commercial paper",
             "borrowing capacity", "covenants", "long-term debt"],
}
SIGNATURE = ("SIGNATURES Pursuant to the requirements of the Securities Exchange Act the registrant has duly caused this report "
             "to be signed. /s/ Jane Roe Principal Executive Officer /s/ John Doe Principal Financial Officer duly authorized")


def topic_text(topic: str, rng: np.random.Generator, n: int = 6) -> str:
    phrases = rng.choice(TOPICS[topic], size=n)
    return " ".join(f"The company discussed {p} during the period and {p} changed compared with the prior year." for p in phrases)


def synthetic_chunks(n_groups: int = 10, docs_per_group: int = 2, seed: int = 0, prefix: str = "AW") -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_groups):
        for d in range(docs_per_group):
            doc = f"DOC_{prefix}_{g:02d}_{d}"
            texts = [topic_text(t, rng) for t in TOPICS for _ in range(2)] + [SIGNATURE]
            for i, t in enumerate(texts):
                out.append({"chunk_id": f"{prefix}_{g:02d}_{d}_{i:02d}", "document_id": doc, "text": t,
                            "metadata": {"company": f"Company{g:02d}", "industry": "Industrials" if g % 2 else "Services",
                                         "source": "report"}})
    return out


BASIS = {t: np.random.default_rng(i + 1).normal(size=DIM) for i, t in enumerate(TOPICS)}


def fake_vector(text: str) -> np.ndarray:
    v = np.zeros(DIM)
    low = text.lower()
    for t, phrases in TOPICS.items():
        v += sum(low.count(p) for p in phrases) * BASIS[t]
    h = np.frombuffer(hashlib.sha256(text.encode()).digest() * 6, dtype=np.uint8)[:DIM].astype(float)
    v += (h / 255.0 - 0.5) * (0.15 * (np.linalg.norm(v) or 1.0))
    return v.astype(np.float32)


class FakeEmbeddings:
    def __init__(self):
        self.calls, self.texts = 0, []

    def __call__(self, batch):
        self.calls += 1; self.texts += batch
        return [fake_vector(t) for t in batch], sum(len(t) // 4 for t in batch)


class FakeLLM:
    """Deterministic stand-in for the chat API. Names come from the payload's distinctive terms."""

    def __init__(self):
        self.calls = []

    def __call__(self, body):
        name = body["response_format"]["json_schema"]["name"]
        user = body["messages"][1]["content"]
        self.calls.append(name)
        if name.startswith("cluster_semantic_review"):
            terms = re.search(r"Most distinctive terms: (.*)", user).group(1).split(", ")
            admin = any(w in user for w in ("SIGNATURES", "/s/")) or bool(re.search(r"\) Page \d+ of \d+", user))   # page footers
            generic = {"company", "discussed", "period", "compared", "prior", "year", "changed"}
            name = next((t for t in terms if not set(t.split()) & generic), terms[0]).title()
            out = {"decision": "REJECT_ADMINISTRATIVE" if admin else "ACCEPT", "coherent": True, "category_name": name,
                   "description": f"Content about {name.lower()}.", "scope": "NONE" if admin else "GLOBAL", "industry": None, "company": None,
                   "semantic_or_structural": "SEMANTIC", "duplicate_of": None, "confidence": 0.9, "reason": "fake"}
        elif name.startswith("broad_family_consolidation"):
            ids = sorted(set(re.findall(r"^(CAT\d{3}) \|", user, re.M)))
            out = {"families": [{"family_id": "BF01", "family_name": "All Topics", "description": "Everything.", "confidence": 0.9,
                                 "member_ids": ids}], "possible_misfits": [], "possible_duplicates": [], "parent_child_notes": []}
        elif name.startswith("novelty_cluster_review"):
            out = {"coherent": True, "fits_existing": False, "existing_category_id": None, "new_category_name": "Novel Topic",
                   "description": "A new topic.", "scope": "GLOBAL", "confidence": 0.8, "reason": "fake"}
        else:
            raise AssertionError(name)
        return {"model": "fake", "choices": [{"message": {"content": json.dumps(out)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
