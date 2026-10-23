"""Versioned wave evolution with deterministic reviewers (no LLM): pool accumulation, cluster decisions, MAP_EXISTING /
CREATE_NEW / KEEP_UNRESOLVED application, new categories start PROBATIONARY, vN unchanged, vN+1 loadable with provenance,
explicit promotion. Neutral corpus (recipes / software / travel + a new 'gardening' topic)."""

import hashlib

import numpy as np
import pytest

from taxonomy_classifier import TaxonomyEngine
from taxonomy_classifier.evolution.promote import promote
from taxonomy_classifier.taxonomy import maturity

from .test_generic_corpus import BowEmbeddings, corpus, generic  # noqa: F401

pytestmark = pytest.mark.slow

GARDEN = ["prune the rose bushes", "water the tomato seedlings", "spread garden compost", "plant the tulip bulbs",
          "weed the vegetable bed", "mulch around the shrubs", "harvest the runner beans", "rake the autumn leaves"]
NOISE = ["zq", "vx", "kj", "wq", "xz", "jq", "qv", "zx"]


def wave_docs(seed, n_docs=6, prefix="W"):
    rng = np.random.default_rng(seed); out = []
    for d in range(n_docs):
        doc = f"{prefix}_garden_{d}"
        for i in range(4):
            text = " ".join(f"First {p}, then {p} again." for p in rng.choice(GARDEN, size=6))
            out.append({"chunk_id": f"{doc}_{i}", "document_id": doc, "text": text, "metadata": {"site": f"site{d % 3}"}})
    return out


class Reviewer:
    """Deterministic stand-in: gardening clusters are new GLOBAL topics; anything else maps to its top candidate."""

    def __init__(self):
        self.calls = []

    def review(self, ctx):
        self.calls.append(ctx["cluster_id"])
        if any(w in ctx["chunks"] for w in ("garden", "tomato", "rose", "tulip", "shrubs", "beans", "leaves", "compost")):
            fit = next((c["category_id"] for c in ctx["candidates"] if c["name"] == "Gardening tasks"), None)
            if fit:                                          # like the real reviewer: an existing candidate fits
                return {"status": "OK", "call_id": f"fake-{ctx['cluster_id']}", "parsed": {"coherent": True, "fits_existing": True,
                        "existing_category_id": fit, "new_category_name": None, "description": None, "scope": "GLOBAL"}}
            return {"status": "OK", "call_id": f"fake-{ctx['cluster_id']}", "parsed": {"coherent": True, "fits_existing": False,
                    "existing_category_id": None, "new_category_name": "Gardening tasks", "description": "Garden work.", "scope": "GLOBAL"}}
        return {"status": "OK", "call_id": "fake", "parsed": {"coherent": False, "fits_existing": False, "existing_category_id": None,
                                                             "new_category_name": None, "description": None, "scope": "GLOBAL"}}


class Structural:
    def review(self, ctx):
        return {"classification": "SEMANTIC_TOPIC", "reason": "fake", "call_id": "fake"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def waved(generic):  # noqa: F811
    cfg = {**generic["cfg"], "novelty": {**generic["cfg"]["novelty"], "hdbscan_settings": [[5, 3], [8, 3]], "umap": {**generic["cfg"]["novelty"]["umap"], "n_neighbors": 8}},
           "evolution": {**generic["cfg"].get("evolution", {}), "creation": {"min_documents": 2, "min_groups": 2, "scopes": ["GLOBAL", "INDUSTRY"]}}}
    store = generic["store"]
    v1_file = store.path("my_custom_corpus", "v1") / "final_taxonomy.json"; v1_sha = sha(v1_file)
    eng = TaxonomyEngine.load("my_custom_corpus", cfg, store, version="v1", embed_transport=BowEmbeddings())
    rv = Reviewer()
    w2 = eng.run_wave(wave_docs(1), wave=2, new_version="v2", reviewer=rv, structural_reviewer=Structural(), log=lambda m: None)
    return {"eng": eng, "store": store, "w2": w2, "v1_sha": v1_sha, "v1_file": v1_file, "rv": rv, "cfg": cfg}


def test_unresolved_accumulate_and_clusters_are_decided(waved):
    w = waved["w2"]
    assert w.summary["pool_before_novelty"] >= 20                       # the new topic is unresolved under v1
    assert len(w.decisions) >= 1 and set(w.decisions.decision) <= {"CREATE_NEW", "MAP_EXISTING", "KEEP_UNRESOLVED", "STRUCTURAL_OR_ADMIN"}


def test_create_new_applies_and_starts_probationary(waved):
    w, store = waved["w2"], waved["store"]
    assert w.new_version == "v2" and "CREATE_NEW" in set(w.decisions.decision)
    v2 = store.load("my_custom_corpus", "v2")
    new = [c for c in v2.categories if c["category_id"] in w.summary["categories_added"]]
    assert new and all(c["status"] == maturity.PROBATIONARY and c["origin"] == "WAVE_2" for c in new)
    assert all(c["created"]["source_residual_cluster"].startswith("W2-R") for c in new)
    labelled = [l for l in w.labels.values() if l["source"] == "NOVELTY_NEW_CATEGORY"]
    assert labelled and all(l["category_id"] in w.summary["categories_added"] for l in labelled)
    assert w.summary["categorised_after_wave"] > w.summary["assigned_at_ingest"]


def test_vN_unchanged_and_vN1_loads_with_provenance(waved):
    store = waved["store"]
    assert sha(waved["v1_file"]) == waved["v1_sha"]                      # v1 bytes untouched
    v1, v2 = store.load("my_custom_corpus", "v1"), store.load("my_custom_corpus", "v2")
    assert len(v2.categories) == len(v1.categories) + len(waved["w2"].summary["categories_added"])
    assert v2.final["derived_from"] == {"taxonomy_id": "my_custom_corpus", "taxonomy_version": "v1",
                                        "final_taxonomy_sha256": waved["v1_sha"]}
    assert v2.final["evolution"]["wave"] == 2
    assert [r["taxonomy_version"] for r in store.list()] == ["v1", "v2"]
    assert (store.path("my_custom_corpus", "v2") / "evolution" / "decisions.csv").exists()
    assert waved["eng"].taxonomy.version == "v2"                         # engine re-bound to the new version


def test_keep_unresolved_and_pool_carries_over(waved):
    w = waved["w2"]
    assert all(r["taxonomy_chunk_id"] not in w.labels for r in w.pool)
    assert w.summary["pool_after"] == len(w.pool)
    recs = w.records
    assert all(r["taxonomy_version"] in ("v1", "v2") for r in recs)


def test_map_existing_into_probationary_category_next_wave(waved):
    """Wave 3: more gardening text, unresolved at ingest (its words are outside the frozen BM25 vocabulary, so the calibrated
    policy cannot reach the new category). The residual cluster is MAPPED to the PROBATIONARY v2 category, not duplicated."""
    eng = waved["eng"]
    w3 = eng.run_wave(wave_docs(7, n_docs=6, prefix="W3") + corpus(n_docs=1, chunks_per_doc=2, seed=5, prefix="W3B"), wave=3, new_version="v3",
                      pool=waved["w2"].pool, reviewer=waved["rv"], structural_reviewer=Structural(), log=lambda m: None)
    garden_new = waved["w2"].summary["categories_added"]
    g_labels = [l for t, l in w3.labels.items() if t.startswith("W3_garden")]
    assert g_labels and all(l["category_id"] in garden_new for l in g_labels)
    assert "MAP_EXISTING" in set(w3.decisions.decision) and {l["source"] for l in g_labels} <= {"NOVELTY_VERIFIED_MAPPING", "HYBRID_RRF"}
    assert not w3.summary["categories_added"] or "Gardening tasks" not in [
        c["name"] for c in eng.store.load("my_custom_corpus", w3.new_version).categories if c["category_id"] in w3.summary["categories_added"]]


def test_explicit_promotion_creates_new_version(waved):
    store = waved["store"]
    v2 = store.load("my_custom_corpus", "v2")
    cid = waved["w2"].summary["categories_added"][0]
    v2p = promote(v2, store, [cid], "v2-promoted", reason="reviewed")
    assert v2p._by_id()[cid]["status"] == maturity.MATURE and v2p.final["derived_from"]["taxonomy_version"] == "v2"
    assert store.load("my_custom_corpus", "v2")._by_id()[cid]["status"] == maturity.PROBATIONARY
    with pytest.raises(ValueError):
        promote(v2p, store, [cid], "v2-again", reason="already mature")
