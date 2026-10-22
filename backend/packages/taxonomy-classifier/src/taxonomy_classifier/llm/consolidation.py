"""API-driven broad-family consolidation + deterministic validation.

1. One call (broad_family_consolidation_v1) with every fine category's summary.
2. Validation: unknown ids are dropped (recorded); ids missing or placed in several families are sent to ONE explicit,
   logged repair call (broad_family_repair_v1) restricted to the existing families. Anything still unresolved ->
   ConsolidationFailure (the pipeline stops; nothing is filled in).
3. Misfits, duplicates and parent/child notes are RECORDED only. Nothing is moved automatically.
4. Deterministic check: parent and child in different families -> recorded as PARENT_CHILD_FAMILY_SPLIT.
"""

from __future__ import annotations

import numpy as np

from .client import LLMClient


class ConsolidationFailure(RuntimeError):
    pass


def category_block(c, nearest, n_ex, chars, text_of, nouns=("documents", "source groups")):
    ex = " || ".join(text_of(t)[:chars] for t in c["representative_chunk_ids"][:n_ex])
    scope = c["scope"] + (f" ({c['industry']})" if c.get("industry") else "") + (f" ({c['company']})" if c.get("company") else "")
    return (f"{c['category_id']} | {c['name']} | scope: {scope} | {c['member_count']} chunks, {c['document_count']} {nouns[0]}, "
            f"{c['group_count']} {nouns[1]} | parent: {c['parent_category_id'] or '-'}\n"
            f"  description: {c['description']}\n  terms: {', '.join(c['top_terms'][:8])}\n"
            f"  excerpts: {ex}\n  nearest: {', '.join(nearest)}")


def consolidate(cats: list[dict], cfg: dict, run_dir, ledger, text_of, transport=None) -> dict:
    cc = cfg["consolidation"]
    nouns = (cc.get("document_noun", "documents"), cc.get("group_noun", "source groups"))
    ids = [c["category_id"] for c in cats]
    C = np.array([c["dense_prototype"] for c in cats], np.float32)
    S = C @ C.T
    blocks = []
    for i, c in enumerate(cats):
        nn = [cats[j]["category_id"] + " " + cats[j]["name"] for j in np.argsort(-S[i]) if j != i][: cc["nearest_categories"]]
        blocks.append(category_block(c, nn, cc["excerpts_per_category"], cc["chars_per_excerpt"], text_of, nouns))
    cli = LLMClient(cfg, run_dir, ledger, "broad_family_consolidation", cc["prompt_version"], cc["reasoning_effort"], cc["max_output_tokens"], transport)
    rec = cli.call(cli.render(n_categories=len(cats), categories="\n\n".join(blocks)), "all_categories")
    if rec["status"] != "OK":
        raise ConsolidationFailure(f"consolidation call {rec['request_id']} status {rec['status']}: {rec.get('validation_errors')}")
    p = rec["parsed"]
    fams = {f["family_id"]: {**f, "member_ids": list(f["member_ids"])} for f in p["families"]}
    seen = {}
    for fid, f in fams.items():
        for i in f["member_ids"]:
            seen.setdefault(i, []).append(fid)
    unknown = sorted(i for i in seen if i not in ids)
    missing = [i for i in ids if i not in seen]
    multi = [i for i in ids if len(seen.get(i, [])) > 1]
    for f in fams.values():
        f["member_ids"] = [i for i in f["member_ids"] if i in ids and i not in multi]
    repair_rec = None
    if missing or multi:
        rc = LLMClient(cfg, run_dir, ledger, "broad_family_repair", cc["repair_prompt_version"], cc["reasoning_effort"], 8000, transport)
        fam_lines = "\n".join(f"{fid} | {f['family_name']} | {f['description']}" for fid, f in fams.items())
        bl = {c["category_id"]: b for c, b in zip(cats, blocks)}
        problem = "missing or assigned to several families: " + ", ".join(f"{i}->{seen.get(i, ['none'])}" for i in missing + multi)
        repair_rec = rc.call(rc.render(families=fam_lines, problem=problem, categories="\n\n".join(bl[i] for i in missing + multi)), "repair")
        if repair_rec["status"] != "OK":
            raise ConsolidationFailure(f"repair call failed: {repair_rec['request_id']}")
        for a in repair_rec["parsed"]["assignments"]:
            if a["category_id"] in missing + multi and a["family_id"] in fams:
                fams[a["family_id"]]["member_ids"].append(a["category_id"])
    final = {i: fid for fid, f in fams.items() for i in f["member_ids"]}
    still = [i for i in ids if i not in final]
    dupl = [i for i in ids if sum(i in f["member_ids"] for f in fams.values()) > 1]
    if still or dupl:
        raise ConsolidationFailure(f"validation failed after repair: missing {still}, multiple {dupl}")
    fams = {k: v for k, v in fams.items() if v["member_ids"]}
    parent = {c["category_id"]: c["parent_category_id"] for c in cats}
    splits = [{"child_id": c, "parent_id": parent[c], "child_family": final[c], "parent_family": final[parent[c]]}
              for c in ids if parent[c] and final[c] != final[parent[c]]]
    return {"families": fams, "primary_family": final, "consolidation_call_id": rec["request_id"],
            "repair_call_id": repair_rec["request_id"] if repair_rec else None,
            "validation": {"unknown_ids_dropped": unknown, "missing_before_repair": missing, "multi_before_repair": multi},
            "possible_misfits": p["possible_misfits"], "possible_duplicates": p["possible_duplicates"],
            "parent_child_notes": p["parent_child_notes"], "parent_child_family_splits": splits}
