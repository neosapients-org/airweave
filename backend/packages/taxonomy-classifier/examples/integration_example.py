"""Minimal library usage: classify exported Airweave chunks and turn the result into Vespa partial updates.

    python examples/integration_example.py artifacts/taxonomy config/taxonomy.yaml examples/airweave_chunks.example.jsonl
"""

import json
import sys

from taxonomy_classifier.integration.airweave import TaxonomyClassifier
from taxonomy_classifier.schemas import read_jsonl


def vespa_partial_update(enriched: dict) -> dict:
    t = enriched["taxonomy"]
    return {"id": enriched["metadata"].get("vespa_id", enriched["chunk_id"]), "fields": {
        "topic_ids": {"assign": t["topic_ids"]}, "topics": {"assign": t["topic_names"]},
        "broad_family_ids": {"assign": t["broad_family_ids"]}, "broad_families": {"assign": t["broad_family_names"]},
        "taxonomy_status": {"assign": t["status"]}, "taxonomy_version": {"assign": t["taxonomy_version"]}}}


if __name__ == "__main__":
    taxonomy_dir, config, chunks_path = sys.argv[1:4]
    airweave_chunks = read_jsonl(chunks_path)            # [{"chunk_id", "document_id", "text", "metadata"?}, ...]
    clf = TaxonomyClassifier(taxonomy_dir, config)       # loads the frozen artifacts once
    result = clf.classify(airweave_chunks)               # 0 LLM calls
    for e in result.enriched[:5]:
        print(json.dumps(vespa_partial_update(e)))
    print(f"{len(result.enriched)} chunks classified; {len(result.unresolved_children)} unresolved children for novelty maintenance",
          file=sys.stderr)
