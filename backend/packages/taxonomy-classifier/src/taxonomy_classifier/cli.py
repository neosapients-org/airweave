"""Command-line entry points (also wrapped by scripts/*.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .schemas import iter_jsonl, read_jsonl, write_jsonl


def build_main(argv=None):
    ap = argparse.ArgumentParser(description="Build (discover) a taxonomy from Airweave chunks. Offline job; makes embedding and LLM calls.")
    ap.add_argument("--input", required=True, help="JSONL of Airweave chunks: chunk_id, document_id, text, metadata (optional)")
    ap.add_argument("--config", required=True, help="YAML config (see config/taxonomy.example.yaml)")
    ap.add_argument("--output", required=True, help="output directory for the frozen taxonomy artifacts")
    ap.add_argument("--no-classify-input", action="store_true", help="do not write build_enriched_chunks.jsonl for the build corpus")
    a = ap.parse_args(argv)
    from .pipeline import build_taxonomy
    m = build_taxonomy(read_jsonl(a.input), load_config(a.config), a.output, classify_input=not a.no_classify_input)
    print(json.dumps({"taxonomy": str(Path(a.output) / "final_taxonomy.json"), **m["result"],
                      "cost_usd": round(m["api"]["cost_ledger"]["total_usd"], 4)}, indent=1))


def classify_main(argv=None):
    ap = argparse.ArgumentParser(description="Classify Airweave chunks against a frozen taxonomy (dense + BM25 + RRF; 0 LLM calls).")
    ap.add_argument("--taxonomy", required=True, help="final_taxonomy.json or the taxonomy directory")
    ap.add_argument("--input", required=True, help="JSONL of Airweave chunks: chunk_id, document_id, text, metadata (optional)")
    ap.add_argument("--output", required=True, help="JSONL of enriched chunks")
    ap.add_argument("--config", default="config/taxonomy.yaml", help="YAML config (embedding + chunking must match the build)")
    ap.add_argument("--unresolved-output", default=None,
                    help="append substantive UNRESOLVED taxonomy chunks to this JSONL (novelty pool for run_novelty_maintenance)")
    ap.add_argument("--batch-size", type=int, default=2000, help="Airweave chunks per batch")
    a = ap.parse_args(argv)
    from .integration.airweave import TaxonomyClassifier
    clf = TaxonomyClassifier(a.taxonomy, a.config)
    out, pool, batch, statuses = [], 0, [], {}

    def flush():
        nonlocal pool
        if not batch:
            return
        r = clf.classify(batch)
        out.extend(r.enriched)
        for e in r.enriched:
            statuses[e["taxonomy"]["status"]] = statuses.get(e["taxonomy"]["status"], 0) + 1
        if a.unresolved_output and r.unresolved_children:
            Path(a.unresolved_output).parent.mkdir(parents=True, exist_ok=True)
            with open(a.unresolved_output, "a") as f:
                for row in r.unresolved_children:
                    f.write(json.dumps(row) + "\n")
            pool += len(r.unresolved_children)
        batch.clear()
    for row in iter_jsonl(a.input):
        batch.append(row)
        if len(batch) >= a.batch_size:
            flush()
    flush()
    write_jsonl(out, a.output)
    print(json.dumps({"enriched_chunks": len(out), "status_counts": statuses, "unresolved_taxonomy_chunks_appended": pool,
                      "embedding": clf.embedder.stats, "llm_calls": 0, "output": a.output}, indent=1))


def novelty_main(argv=None):
    ap = argparse.ArgumentParser(description="Novelty maintenance: cluster the unresolved pool, one LLM review per recurring residual "
                                             "cluster, write mapping / new-category PROPOSALS (the frozen taxonomy is not modified).")
    ap.add_argument("--taxonomy", required=True, help="final_taxonomy.json or the taxonomy directory")
    ap.add_argument("--input", required=True, help="unresolved_chunks.jsonl written by classify --unresolved-output")
    ap.add_argument("--config", default="config/taxonomy.yaml")
    ap.add_argument("--output", default=None, help="output directory (default: <taxonomy dir>/novelty/<timestamp>)")
    a = ap.parse_args(argv)
    from datetime import datetime, timezone

    from .novelty.maintenance import run_maintenance
    from .taxonomy import artifacts
    cfg = load_config(a.config)
    if not cfg["novelty"].get("enabled", True):
        print("novelty.enabled is false in the config; nothing to do"); return
    tax = artifacts.load(a.taxonomy)
    out = a.output or str(tax.path / "novelty" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    s = run_maintenance(tax, read_jsonl(a.input), cfg, out)
    print(json.dumps({k: v for k, v in s.items() if k != "new_categories"} | {"new_categories": len(s["new_categories"]), "output": out}, indent=1))


if __name__ == "__main__":
    sys.exit({"build": build_main, "classify": classify_main, "novelty": novelty_main}[sys.argv.pop(1)]())
