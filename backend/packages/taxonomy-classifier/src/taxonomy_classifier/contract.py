"""Stable integration contract: one record per Airweave / retrieval chunk with a `topics[]` LIST. Stable API.

    from taxonomy_classifier.contract import classify
    result = classify(airweave_chunks, taxonomy="artifacts/taxonomy", config="config/taxonomy.yaml")
    for rec in result.records:
        for t in rec["topics"]:            # 0, 1 or many topics; never a single "category" field
            t["category_id"], t["category_name"], t["support_share"]

`topics[]` may hold 0, 1 or many entries. Today each ~300-token taxonomy child carries at most the categories the
assigner accepts for it, and a parent chunk gets one topic per distinct category among its children. The multi-label
phase changes how many categories a child may carry; it does not change this record shape.

Field names follow enriched schema v2 (`taxonomy_coverage`, `ancestor_category_ids`, `category_depth`,
`supporting_taxonomy_chunks`, `support_share`) so research outputs and integration outputs read the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import CHILD_UNRESOLVED, NON_SUBSTANTIVE_CHILD

CONTRACT_VERSION = "topics/1"          # bumped only when a field is removed or re-typed; additive fields keep it
TOPIC_STATUS_ACCEPTED = "accepted"
MEMBERSHIP_ONLY_SOURCE = "DISCOVERY_CLUSTER_MEMBER"   # build corpus only: label from discovery-cluster membership


@dataclass
class ContractResult:
    records: list[dict]
    unresolved_children: list[dict] = field(default_factory=list)   # novelty pool input
    embedding_stats: dict = field(default_factory=dict)


def _hierarchy(taxonomy) -> dict[str, dict]:
    """category_id -> {name, parent_id, family_id, family_name} from a built Taxonomy or a CanonicalTaxonomy."""
    out = {}
    for c in taxonomy.categories:
        out[c["category_id"]] = {"name": c["name"], "parent_id": c.get("parent_id", c.get("parent_category_id")),
                                 "family_id": c.get("broad_family_id"), "status": c.get("status", "MATURE"),
                                 "family_name": c.get("broad_family_name", c.get("broad_family"))}
    return out


def _ancestors(cid: str, h: dict) -> list[str]:
    out, p, seen = [], h.get(cid, {}).get("parent_id"), {cid}
    while p and p in h and p not in seen:
        seen.add(p); out.append(p); p = h[p]["parent_id"]
    return out


def taxonomy_coverage(children: list[dict], topics: list[dict]) -> dict:
    """Same fields and rules as enriched schema v2 TaxonomyCoverage."""
    total = len(children)
    supporting = {c for t in topics for c in t["supporting_taxonomy_chunks"]}
    categorised = len(supporting)
    topic_ids = {t["category_id"] for t in topics}
    by_assignment = sum(1 for c in children if c["taxonomy_chunk_id"] in supporting and any(
        src != MEMBERSHIP_ONLY_SOURCE for k, src in c.get("topic_sources", {k: "HYBRID_RRF" for k in c["topic_ids"]}).items()
        if k in topic_ids))
    return {"taxonomy_children_total": total, "taxonomy_children_categorised": categorised,
            "taxonomy_children_uncategorised": total - categorised,
            "taxonomy_children_administrative": sum(1 for c in children if c["status"] in NON_SUBSTANTIVE_CHILD),
            "taxonomy_children_categorised_by_assignment_only": by_assignment,
            "child_coverage_ratio": round(categorised / total, 4) if total else 0.0, "topic_count": len(topics),
            "coverage_status": "NONE" if categorised == 0 else "FULL" if categorised == total else "PARTIAL"}


def to_contract(enriched: dict, hierarchy=None) -> dict:
    """Convert one enriched record (integration.aggregation.aggregate output) into the stable contract record.

    hierarchy: the built Taxonomy or a CanonicalTaxonomy, used for parent / ancestors / depth. Without it those
    fields are null / [] / 1 only for categories it does not know.
    """
    tx = enriched["taxonomy"]
    h = _hierarchy(hierarchy) if hierarchy is not None else {}
    topics = []
    for t in tx["topics"]:
        cid = t["category_id"]
        anc = _ancestors(cid, h)
        topics.append({
            "category_id": cid, "category_name": t["name"], "confidence": None, "status": TOPIC_STATUS_ACCEPTED,
            "category_status": h.get(cid, {}).get("status"),
            "broad_family_id": t.get("broad_family_id"), "broad_family_name": t.get("broad_family_name"),
            "parent_category_id": anc[0] if anc else None,
            "parent_category_name": h[anc[0]]["name"] if anc else None,
            "ancestor_category_ids": anc, "ancestor_category_names": [h[a]["name"] for a in anc],
            "category_depth": len(anc) + 1,
            "supporting_taxonomy_chunks": list(t["supporting_taxonomy_chunks"]),
            "supporting_children": len(t["supporting_taxonomy_chunks"]),
            "support_share": t["support_share"], "sources": list(t["sources"])})
    return {"contract_version": CONTRACT_VERSION, "chunk_id": enriched["chunk_id"], "document_id": enriched["document_id"],
            "taxonomy_id": tx.get("taxonomy_id"), "taxonomy_version": tx["taxonomy_version"], "status": tx["status"], "topics": topics,
            "assignment": {k: (tx.get("assignment") or {}).get(k) for k in ("acceptance_policy", "acceptance_model", "threshold")},
            "taxonomy_coverage": taxonomy_coverage(tx["taxonomy_chunks"], topics),
            "taxonomy_children": [{"taxonomy_chunk_id": c["taxonomy_chunk_id"], "status": c["status"],
                                   "char_start": c["char_start"], "char_end": c["char_end"],
                                   "topic_ids": list(c["topic_ids"])} for c in tx["taxonomy_chunks"]],
            "content_metadata": enriched.get("content_metadata", {})}


def classify(airweave_chunks, taxonomy, config, hierarchy=None, embed_transport=None) -> ContractResult:
    """Classify Airweave chunks against a frozen built taxonomy and return contract records (0 LLM calls).

    taxonomy: built taxonomy directory (scripts/build_taxonomy.py output) or a loaded Taxonomy.
    hierarchy: optional CanonicalTaxonomy; defaults to the built taxonomy's own parent links.
    """
    from .integration.airweave import TaxonomyClassifier

    clf = TaxonomyClassifier(taxonomy, config, embed_transport)
    res = clf.classify(airweave_chunks)
    h = hierarchy if hierarchy is not None else clf.taxonomy
    return ContractResult([to_contract(e, h) for e in res.enriched], res.unresolved_children, res.embedding_stats)


__all__ = ["CONTRACT_VERSION", "ContractResult", "classify", "to_contract", "taxonomy_coverage", "CHILD_UNRESOLVED"]
