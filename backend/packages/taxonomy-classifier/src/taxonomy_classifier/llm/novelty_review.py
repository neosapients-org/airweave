"""Cluster-level LLM review of a RESIDUAL (unresolved) cluster during novelty maintenance.

One call per recurring residual cluster, never one per chunk. The reviewer sees the cluster's statistics, distinctive
terms, the most similar existing categories (whole-taxonomy dense+BM25 RRF search, done in code) and representative
chunks. It only recommends: novelty/maintenance.py applies per-chunk verification and recurrence gates in code.
"""

from __future__ import annotations

from .client import LLMClient
from .cluster_review import dist


def client(cfg: dict, out_dir, ledger, transport=None) -> LLMClient:
    n = cfg["novelty"]
    return LLMClient(cfg, out_dir, ledger, "novelty_cluster_review", n["prompt_version"], n["reasoning_effort"], n["max_output_tokens"], transport)


def payload(cli: LLMClient, rid: str, sub, terms: list[str], candidates: list[str], name: dict, desc: dict, chunks: str) -> str:
    return cli.render(cluster_id=rid, member_count=len(sub), document_count=sub.document_id.nunique(), company_count=sub.group.nunique(),
                      company_distribution=dist(sub.group), top_terms=", ".join(terms) or "(none)",
                      candidates="\n".join(f"{c} | {name[c]} | {desc[c]}" for c in candidates), chunks=chunks)
