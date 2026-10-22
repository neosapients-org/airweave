"""Airweave adapter: classify Airweave retrieval chunks against a FROZEN taxonomy. Makes 0 LLM calls.

    from taxonomy_classifier.integration.airweave import classify_chunks
    enriched = classify_chunks(airweave_chunks, taxonomy="artifacts/taxonomy", config="config/taxonomy.yaml")

Input: [{"chunk_id", "document_id", "text", "metadata"?}] (exported Airweave chunks, or chunks straight out of
Airweave's SemanticChunker). Output: one enriched record per input chunk, same chunk_id (see schemas.py / README).
Embedding calls are made only for taxonomy children whose text is not in the embedding cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import load_config
from ..assignment.acceptance import make_policy
from ..embeddings.embedder import Embedder
from ..prepare import prepare
from ..schemas import CHILD_UNRESOLVED, parse_chunks
from ..taxonomy import artifacts
from ..taxonomy.models import Taxonomy
from .aggregation import aggregate


class ConfigMismatch(ValueError):
    pass


@dataclass
class ClassificationResult:
    enriched: list[dict]
    unresolved_children: list[dict] = field(default_factory=list)   # novelty pool (substantive UNRESOLVED children)
    embedding_stats: dict = field(default_factory=dict)
    child_trace: list[dict] = field(default_factory=list)            # internal/debug: per-child decision + scores
    provenance: dict = field(default_factory=dict)                   # acceptance policy, model, threshold, taxonomy version
    children: list[dict] = field(default_factory=list)               # child records fed to aggregation (lineage, status, topics)


class TaxonomyClassifier:
    def __init__(self, taxonomy: str | Path | Taxonomy, config: str | Path | dict, embed_transport=None):
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        self.taxonomy = taxonomy if isinstance(taxonomy, Taxonomy) else artifacts.load(taxonomy)
        self._check_compatible()
        self.embedder = Embedder(self.cfg, None, embed_transport)
        self.policy = make_policy(self.taxonomy, self.cfg)

    def _check_compatible(self):
        f, c = self.taxonomy.final, self.cfg
        problems = []
        if f["embedding"] != {"model": c["embedding"]["model"], "dimensions": c["embedding"]["dimensions"]}:
            problems.append(f"embedding {f['embedding']} (taxonomy) vs {c['embedding']['model']}/{c['embedding']['dimensions']} (config)")
        if f["taxonomy_chunking"] != c["taxonomy_chunking"]:
            problems.append("taxonomy_chunking differs from the build")
        if problems:
            raise ConfigMismatch("config is not compatible with the frozen taxonomy: " + "; ".join(problems))

    def classify(self, airweave_chunks) -> ClassificationResult:
        chunks = parse_chunks(airweave_chunks)
        # content-form patterns are frozen with the taxonomy so classification bypasses exactly what the build bypassed
        cfg = {**self.cfg, "content_form": self.taxonomy.final.get("content_form", self.cfg.get("content_form", {}))}
        tc, meta, X, tags = prepare(chunks, cfg, self.embedder)
        B = self.taxonomy.bm25.transform(list(meta.text))
        admin = (tags.admin_quality == "ADMINISTRATIVE").to_numpy()
        res = self.policy.assign(X, B, admin, list(meta.taxonomy_chunk_id), tokens=[t["token_count"] for t in tc],
                                 content_forms=list(tags.content_form))
        prov = {**self.policy.provenance(), "taxonomy_id": self.taxonomy.taxonomy_id, "taxonomy_version": self.taxonomy.version}
        source = "LEARNED_ACCEPTANCE" if self.policy.name == "learned" else "HYBRID_RRF"
        children, pool, trace = [], [], []
        for r, t, g in zip(res, tc, meta.group):
            children.append({**{k: t[k] for k in ("taxonomy_chunk_id", "chunk_id", "char_start", "char_end", "content_form")},
                             "status": r["status"],
                             "topics": [{"category_id": k, "source": source, "score": s} for k, s in zip(r["direct"], r["scores"])]})
            trace.append({"taxonomy_chunk_id": t["taxonomy_chunk_id"], "status": r["status"], "category_ids": list(r["direct"]),
                          "scores": list(r["scores"]), **r.get("trace", {})})
            if r["status"] == CHILD_UNRESOLVED:
                pool.append({k: t[k] for k in ("taxonomy_chunk_id", "chunk_id", "document_id", "text", "text_sha256", "token_count")} |
                            {"group": g, "content_form": t["content_form"], "taxonomy_version": self.taxonomy.version})
        enriched = aggregate(chunks, children, self.taxonomy, cfg)
        for e in enriched:
            e["taxonomy"]["assignment"] = prov
        return ClassificationResult(enriched, pool, dict(self.embedder.stats), trace, prov, children)


def classify_chunks(airweave_chunks, taxonomy, config, embed_transport=None) -> list[dict]:
    """Convenience wrapper: returns the enriched chunks only."""
    return TaxonomyClassifier(taxonomy, config, embed_transport).classify(airweave_chunks).enriched
