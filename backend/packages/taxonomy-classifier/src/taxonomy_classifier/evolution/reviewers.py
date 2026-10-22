"""Pluggable cluster-level reviewers for wave evolution. Any object with `review(ctx) -> dict` works.

NoveltyReviewer.review(ctx) -> {"status": "OK"|..., "parsed": {coherent, fits_existing, existing_category_id,
                                 new_category_name, description, scope, ...}, "call_id": str|None}
StructuralReviewer.review(ctx) -> {"classification": SEMANTIC_TOPIC|STRUCTURAL_OR_DOCUMENT_FORM|ADMINISTRATIVE|UNCLEAR,
                                   "reason": str, "call_id": str|None}
The LLM implementations make one logged, cached, schema-validated call per cluster (llm/client.py). Tests and replays
use deterministic implementations; the wave mechanics never require a live API.
"""

from __future__ import annotations

from ..llm.client import LLMClient


class LLMNoveltyReviewer:
    def __init__(self, cfg: dict, run_dir, ledger, transport=None):
        n = cfg["novelty"]
        self.cli = LLMClient(cfg, run_dir, ledger, "novelty_cluster_review", n["prompt_version"], n["reasoning_effort"], n["max_output_tokens"], transport)

    def review(self, ctx: dict) -> dict:
        user = self.cli.render(cluster_id=ctx["cluster_id"], member_count=ctx["member_count"], document_count=ctx["document_count"],
                               company_count=ctx["group_count"], company_distribution=ctx["group_distribution"], top_terms=", ".join(ctx["top_terms"]),
                               candidates="\n".join(f"{c['category_id']} | {c['name']} | {c['description']}" for c in ctx["candidates"]) or "(none)",
                               chunks=ctx["chunks"])
        call = self.cli.call(user, ctx["cluster_id"], {"candidate_category_ids": [c["category_id"] for c in ctx["candidates"]]})
        return {"status": call["status"], "parsed": call["parsed"], "call_id": call["request_id"]}


class LLMStructuralReviewer:
    def __init__(self, cfg: dict, run_dir, ledger, transport=None):
        s = cfg.get("evolution", {}).get("structural_filter", {})
        self.cli = LLMClient(cfg, run_dir, ledger, "structural_category_review", s.get("prompt_version", "structural_category_review_v2"),
                             s.get("reasoning_effort", "low"), s.get("max_output_tokens", 1500), transport)

    def review(self, ctx: dict) -> dict:
        user = self.cli.render(name=ctx["name"], description=ctx["description"], terms=", ".join(ctx["terms"]), chunks=ctx["chunks"])
        call = self.cli.call(user, ctx["name"][:60].replace(" ", "_"))
        q = call["parsed"] or {}
        return {"classification": q.get("classification", "CALL_" + call["status"]), "reason": q.get("reason"), "call_id": call["request_id"]}
