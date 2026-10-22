"""Import test + guard against corpus-specific assumptions in runtime code."""

import re

from .conftest import ROOT

FORBIDDEN = ["SEC", "10-K", "10-Q", "Apple", "JPMorgan", "Visa", "Walmart", "Pfizer", "Alphabet", "Caterpillar",
             "Experiment 9", "Experiment 12", "Experiment 13", "ticker", "filing"]


def test_imports():
    import taxonomy_classifier
    from taxonomy_classifier import TaxonomyClassifier, build_taxonomy, classify_chunks, load_taxonomy  # noqa: F401
    assert taxonomy_classifier.__version__


def test_runtime_code_and_prompts_are_domain_agnostic():
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "src").rglob("*.txt"))
    hits = [(f.name, w) for f in files for w in FORBIDDEN if re.search(rf"\b{re.escape(w)}\b", f.read_text())]
    assert not hits, hits
