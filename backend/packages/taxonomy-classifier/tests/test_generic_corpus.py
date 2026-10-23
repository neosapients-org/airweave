"""Architectural smoke test: a corpus unrelated to finance, no pre-existing taxonomy, through the whole engine.

new corpus -> taxonomy children -> discovery -> artifact saved in a store -> reloaded -> classified -> novelty proposals.
Offline: a bag-of-words fake embedder and the deterministic fake reviewer (no API calls, no spend). This checks that the
engine runs on arbitrary data, not taxonomy quality.
"""

import copy
import hashlib

import numpy as np
import pytest

from taxonomy_classifier import LocalTaxonomyStore, TaxonomyEngine

from .conftest import FakeLLM

pytestmark = pytest.mark.slow

DIM = 64
VOCAB = {
    "recipes": ["whisk the eggs", "preheat the oven", "simmer the sauce", "knead the dough", "fold in the flour",
                "season with salt", "chop the onions", "bake until golden"],
    "software": ["install the package", "run the command line", "configure the environment variable", "call the api endpoint",
                 "read the error log", "upgrade the dependency", "write a unit test", "deploy the container"],
    "travel": ["book the train ticket", "visit the old town", "pack a light backpack", "check in at the hostel",
               "walk along the harbour", "rent a bicycle", "catch the ferry", "explore the mountain trail"],
}
FINANCE_WORDS = ("tax", "debt", "fair value", "revenue", "liquidity", "pension", "insurance", "filing", "10-q", "10-k")


def bow_vector(text: str) -> np.ndarray:
    """Sum of per-word pseudo-random vectors: similar vocabulary -> similar vectors, for any language or domain."""
    v = np.zeros(DIM)
    for w in text.lower().split():
        seed = int.from_bytes(hashlib.sha256(w.strip(".,").encode()).digest()[:4], "little")
        v += np.random.default_rng(seed).normal(size=DIM)
    return v.astype(np.float32)


class BowEmbeddings:
    def __call__(self, batch):
        return [bow_vector(t) for t in batch], sum(len(t) // 4 for t in batch)


def corpus(n_docs=6, chunks_per_doc=4, seed=0, prefix="G"):
    rng = np.random.default_rng(seed)
    out = []
    for topic, phrases in VOCAB.items():
        for d in range(n_docs):
            doc = f"{prefix}_{topic}_{d}"
            for i in range(chunks_per_doc):
                text = " ".join(f"First {p}, then {p} again." for p in rng.choice(phrases, size=6))
                out.append({"chunk_id": f"{doc}_{i}", "document_id": doc, "text": text, "metadata": {"site": f"site{d % 4}"}})
            out.append({"chunk_id": f"{doc}_footer", "document_id": doc, "text": f"Page {d} of 9", "metadata": {"site": f"site{d % 4}"}})
    return out


@pytest.fixture(scope="module")
def generic(tmp_path_factory, request):
    import yaml
    from .conftest import ROOT
    tmp = tmp_path_factory.mktemp("generic")
    c = copy.deepcopy(yaml.safe_load(open(ROOT / "config" / "taxonomy.example.yaml")))
    c["input"]["fields"] = {"group": "site"}                    # corpus-specific metadata key; nothing else configured
    c["content_form"]["patterns"] = {}                         # no SEC boilerplate rules for this corpus
    c["embedding"].update({"model": "fake-bow", "dimensions": DIM, "cache_path": str(tmp / "emb.sqlite")})
    c["llm"].update({"cache_dir": str(tmp / "llm_cache"), "corpus_description": "recipes, software docs and travel notes"})
    c["clustering"]["hdbscan_settings"] = [[10, 5], [15, 5]]
    c["clustering"]["umap"]["n_neighbors"] = 10
    c["clustering"]["screens"].update({"min_documents": 3})
    c["assignment"]["bm25"]["min_df"] = 2
    store = LocalTaxonomyStore(tmp / "store")
    llm = FakeLLM()
    engine = TaxonomyEngine(c, store=store, embed_transport=BowEmbeddings(), llm_transport=llm)
    tax = engine.discover(corpus(), taxonomy_id="my_custom_corpus", domain="household_notes", version="v1", log=lambda m: None)
    return {"engine": engine, "store": store, "tax": tax, "cfg": c, "tmp": tmp, "llm": llm}


def test_discovery_builds_and_saves_a_versioned_artifact(generic):
    tax = generic["tax"]
    assert len(tax.categories) >= 2
    assert (tax.taxonomy_id, tax.domain, tax.version, tax.schema_version) == ("my_custom_corpus", "household_notes", "v1",
                                                                             "taxonomy-artifact/1")
    assert tax.final["created_at"] and tax.final["source"]["documents"] == 18
    listed = generic["store"].list()
    assert [(r["taxonomy_id"], r["taxonomy_version"]) for r in listed] == [("my_custom_corpus", "v1")]
    cal = tax.final["calibration"]
    assert cal["junk_pool"] > 0 and cal["reject_prototypes"] >= 1 and cal["dense_floor"] < 0.95   # footers became reject evidence
    names = " ".join(c["name"].lower() for c in tax.categories)
    assert not any(w in names for w in FINANCE_WORDS)


def test_store_reload_keeps_ids_and_versions_are_immutable(generic):
    store, tax = generic["store"], generic["tax"]
    again = store.load("my_custom_corpus")
    assert [c["category_id"] for c in again.categories] == [c["category_id"] for c in tax.categories]
    with pytest.raises(FileExistsError):
        generic["engine"].discover(corpus(), taxonomy_id="my_custom_corpus", version="v1", log=lambda m: None)


def test_reloaded_engine_classifies_new_documents_into_topics(generic):
    engine = TaxonomyEngine.load("my_custom_corpus", generic["cfg"], generic["store"], embed_transport=BowEmbeddings())
    new = corpus(n_docs=1, chunks_per_doc=2, seed=9, prefix="NEW")
    n_llm = len(generic["llm"].calls)
    res = engine.classify(new)
    assert len(generic["llm"].calls) == n_llm                                   # classification makes no LLM calls
    assert [r["chunk_id"] for r in res.records] == [c["chunk_id"] for c in new]
    for r in res.records:
        assert (r["taxonomy_id"], r["taxonomy_version"]) == ("my_custom_corpus", "v1")
        assert isinstance(r["topics"], list) and r["taxonomy_coverage"]["topic_count"] == len(r["topics"])
        assert r["taxonomy_coverage"]["coverage_status"] in ("FULL", "PARTIAL", "NONE")
    content = [r for r in res.records if not r["chunk_id"].endswith("_footer")]
    assert sum(bool(r["topics"]) for r in content) >= len(content) - 1
    footer = next(r for r in res.records if r["chunk_id"].endswith("_footer"))
    assert footer["topics"] == [] and footer["status"] == "ADMINISTRATIVE"


def test_unresolved_text_enters_the_pool_and_evolution_runs(generic):
    engine = generic["engine"]
    off = [{"chunk_id": f"OFF{i}", "document_id": f"OFFDOC{i % 3}", "metadata": {"site": f"site{i % 3}"},
            "text": " ".join(["Quarterly board meeting minutes approved the gardening committee budget and tulip planting schedule."] * 4)}
           for i in range(3)]
    res = engine.classify(off)
    assert res.unresolved_children
    proposal = engine.evolve(res.unresolved_children, generic["tmp"] / "novelty", log=lambda m: None)
    assert proposal["base_taxonomy_version"] == "v1" and proposal["pool_size"] == len(res.unresolved_children)
    assert generic["store"].load("my_custom_corpus").version == "v1"           # proposals never modify the taxonomy
