"""An empty vocabulary (texts that share too few Latin-script words: a small, varied corpus, or one in another script).

The distinctive terms are context for the reviewer, not a prerequisite: an empty vocabulary gives no terms and the run
goes on (it used to abort discovery and the update wave). The keyword index (BM25) does need a vocabulary: a discovery
without one stops with a plain CorpusTooSmall before the pieces are embedded or reviewed (no paid call at all), instead
of the vectorizer's error after the paid embeddings and reviews.
"""

import json
import string

import pytest
from sklearn.feature_extraction.text import CountVectorizer

from taxonomy_classifier.assignment.classifier import CorpusTooSmall
from taxonomy_classifier.discovery.clustering import fit_terms, top_terms
from taxonomy_classifier.pipeline import _terms_vectorizer, build_taxonomy

from .conftest import ROOT, FakeLLM, fake_vector, synthetic_chunks
from .test_generic_corpus import generic  # noqa: F401
from .test_wave_evolution import Structural, wave_docs, waved  # noqa: F401

# Latin letters to Greek ones and back: the texts keep their words, but no Latin term survives
LATIN = string.ascii_lowercase + string.ascii_uppercase
GREEK = "αβγδεζηθικλμνξοπρστυφχψωϊϋ" + "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΪΫ"
TO_GREEK, TO_LATIN = str.maketrans(LATIN, GREEK), str.maketrans(GREEK, LATIN)


def test_texts_that_share_no_terms_give_no_terms_instead_of_failing():
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"]
    cv, TF = fit_terms(_terms_vectorizer(), texts)
    assert (cv, TF) == (None, None)
    assert top_terms(cv, TF, {"CL0000": [0, 1], "CL0001": [2, 3]}) == {"CL0000": [], "CL0001": []}


def test_any_other_vectorizer_error_still_surfaces():
    with pytest.raises(ValueError, match="max_df"):                       # an impossible setting is not an empty corpus
        fit_terms(CountVectorizer(min_df=2, max_df=1), ["alpha beta", "alpha gamma"])


@pytest.mark.slow
def test_a_discovery_without_a_keyword_vocabulary_stops_before_any_paid_call(tmp_path):
    import yaml

    from .conftest import DIM

    chunks = [{**c, "text": c["text"].translate(TO_GREEK)} for c in synthetic_chunks()]
    cfg = yaml.safe_load(open(ROOT / "config" / "taxonomy.example.yaml"))
    cfg["embedding"].update({"model": "fake-embedding", "dimensions": DIM, "cache_path": str(tmp_path / "emb.sqlite")})
    cfg["llm"].update({"cache_dir": str(tmp_path / "llm_cache"), "corpus_description": "synthetic", "max_cost_usd": 1.0})
    cfg["clustering"]["hdbscan_settings"] = [[10, 5], [15, 5]]
    cfg["clustering"]["umap"]["n_neighbors"] = 10
    cfg["assignment"]["bm25"]["min_df"] = 2

    embedded = []

    def embed(batch):
        embedded.append(batch)
        return [fake_vector(t.translate(TO_LATIN)) for t in batch], sum(len(t) // 4 for t in batch)

    llm = FakeLLM()
    with pytest.raises(CorpusTooSmall, match="no keyword vocabulary"):
        build_taxonomy(chunks, cfg, tmp_path / "out", log=lambda m: None, embed_transport=embed, llm_transport=llm)
    assert embedded == [] and llm.calls == []          # stopped before any paid call: nothing embedded, nothing reviewed
    assert json.loads((tmp_path / "out" / "cost_ledger.json").read_text())["total_usd"] == 0


@pytest.mark.slow
def test_an_update_wave_on_text_in_another_script_goes_on(waved):  # noqa: F811
    """The wave keeps the taxonomy's keyword index; only its review terms are fitted, and they may be empty."""
    greek = [{**c, "text": c["text"].translate(TO_GREEK)} for c in wave_docs(11, n_docs=6, prefix="GR")]
    w = waved["eng"].run_wave(greek, wave=9, new_version="v-greek", reviewer=waved["rv"],
                              structural_reviewer=Structural(), log=lambda m: None)
    assert w.summary["pool_before_novelty"] >= 20      # residual clustering (and its term fit) ran
    assert len(w.decisions) >= 1
