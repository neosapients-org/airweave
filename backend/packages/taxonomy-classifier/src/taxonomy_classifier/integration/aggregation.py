"""Aggregate taxonomy-child labels back onto the original Airweave chunk.

A topic is attached to an Airweave chunk if at least `aggregation.min_supporting_children` of its taxonomy children
carry it as a DIRECT label; the supporting children are listed. Broad families follow from the topics.
Content form is aggregated separately (child counts) and never mixed with topics.
"""

from __future__ import annotations

from collections import Counter

from .lineage import chunk_status, group_by_parent


def aggregate(chunks, children: list[dict], taxonomy, cfg: dict) -> list[dict]:
    """chunks: AirweaveChunk list. children: [{taxonomy_chunk_id, chunk_id, status, topics: [{category_id, source, score}],
    content_form, char_start, char_end}]. Returns one enriched record per Airweave chunk (input order)."""
    by_parent = group_by_parent(children)
    k = cfg["aggregation"]["min_supporting_children"]
    out = []
    for a in chunks:
        kids = by_parent.get(a.chunk_id, [])
        sup, src = {}, {}
        for c in kids:
            for t in c["topics"]:
                sup.setdefault(t["category_id"], []).append(c["taxonomy_chunk_id"])
                src.setdefault(t["category_id"], set()).add(t["source"])
        topics = []
        for cid, v in sorted(sup.items(), key=lambda kv: -len(kv[1])):
            if len(v) < k:
                continue
            info = taxonomy.info(cid)
            topics.append({"category_id": cid, "name": info["name"], "broad_family_id": info["broad_family_id"],
                           "broad_family_name": info["broad_family_name"], "supporting_taxonomy_chunks": sorted(v),
                           "support_share": round(len(v) / max(len(kids), 1), 3), "sources": sorted(src[cid])})
        fams = {}
        for t in topics:
            if t["broad_family_id"]:
                fams.setdefault(t["broad_family_id"], t["broad_family_name"])
        forms = Counter(c["content_form"] for c in kids)
        out.append({
            "chunk_id": a.chunk_id, "document_id": a.document_id, "metadata": a.metadata,
            "taxonomy": {
                "taxonomy_id": getattr(taxonomy, "taxonomy_id", None), "taxonomy_version": taxonomy.version,
                "status": chunk_status([c["status"] for c in kids], bool(topics)),
                "topics": topics,
                "topic_ids": [t["category_id"] for t in topics], "topic_names": [t["name"] for t in topics],
                "broad_family_ids": sorted(fams), "broad_family_names": [fams[f] for f in sorted(fams)],
                "taxonomy_chunks": [{"taxonomy_chunk_id": c["taxonomy_chunk_id"], "status": c["status"], "char_start": c["char_start"],
                                     "char_end": c["char_end"], "topic_ids": [t["category_id"] for t in c["topics"]],
                                     "topic_sources": {t["category_id"]: t["source"] for t in c["topics"]}} for c in kids]},
            "content_metadata": {"content_form": forms.most_common(1)[0][0] if forms else None, "content_form_counts": dict(forms)}})
    return out
