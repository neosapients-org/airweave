"""Airweave taxonomy classifier: discover a topic taxonomy from Airweave retrieval chunks, then tag new chunks with
semantic topics and broad families (dense + BM25 + RRF, no LLM at classification time)."""

__version__ = "1.0.0"

__all__ = ["__version__", "TaxonomyEngine", "LocalTaxonomyStore", "classify", "classify_chunks", "TaxonomyClassifier",
           "build_taxonomy", "load_taxonomy", "load_registry", "load_canonical_taxonomy", "validate_taxonomy"]


def __getattr__(name):            # lazy: importing the package does not import UMAP/HDBSCAN
    if name == "TaxonomyEngine":
        from .engine import TaxonomyEngine
        return TaxonomyEngine
    if name == "LocalTaxonomyStore":
        from .store import LocalTaxonomyStore
        return LocalTaxonomyStore
    if name in ("classify_chunks", "TaxonomyClassifier"):
        from .integration import airweave
        return getattr(airweave, name)
    if name == "classify":
        from .contract import classify
        return classify
    if name in ("load_registry", "load_canonical_taxonomy", "validate_taxonomy"):
        from . import canonical
        return getattr(canonical, name)
    if name in ("build_taxonomy", "load_taxonomy"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
