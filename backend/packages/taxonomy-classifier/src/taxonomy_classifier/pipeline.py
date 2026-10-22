"""Taxonomy BUILD (offline, occasional).

Airweave chunks -> ~300-token children -> embeddings -> UMAP/HDBSCAN grid -> stable candidate clusters
-> LLM cluster review -> fine categories -> LLM broad-family consolidation -> calibration -> frozen artifacts.

Decisions: cluster membership comes from discovery/ (code); names, descriptions and accept/reject decisions from logged
LLM responses (llm/); broad families from a logged LLM response plus deterministic validation; thresholds from
calibration code. Nothing is decided interactively.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from . import __version__
from .assignment.classifier import HybridAssigner
from .assignment.sparse import BM25, sparse_profiles
from .config import PROMPTS_DIR, public
from .discovery import candidate_clusters, clustering
from .embeddings.embedder import Embedder
from .integration.aggregation import aggregate
from .llm import cluster_review, consolidation
from .llm.costs import CostLedger
from .prepare import prepare
from .schemas import parse_chunks, write_json, write_jsonl
from .taxonomy import artifacts, builder
from .taxonomy.maturity import MATURE
from .taxonomy.models import Taxonomy
from .taxonomy.representation import category_stats, choose_k, subprotos


def _terms_vectorizer():
    return CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")


def _prompt_versions(cfg):
    return {"semantic_review": cfg["semantic_review"]["prompt_version"], "consolidation": cfg["consolidation"]["prompt_version"],
            "repair": cfg["consolidation"]["repair_prompt_version"], "novelty": cfg["novelty"]["prompt_version"]}


def build_taxonomy(airweave_chunks, cfg: dict, out_dir, log=print, embed_transport=None, llm_transport=None,
                   classify_input: bool = True) -> dict:
    t_start = time.perf_counter()
    out = Path(out_dir); (out / "intermediate").mkdir(parents=True, exist_ok=True)
    chunks = parse_chunks(airweave_chunks)
    ledger = CostLedger(cfg["llm"]["max_cost_usd"], out / "cost_ledger.json")
    timings = {}

    def stage(name, t0):
        timings[name] = round(time.perf_counter() - t0, 2); log(f"== {name}: {timings[name]}s")

    # 1. children + embeddings + content form
    t0 = time.perf_counter()
    emb = Embedder(cfg, ledger, embed_transport)
    tc, meta, X, tags = prepare(chunks, cfg, emb)
    write_jsonl(tc, out / "intermediate" / "taxonomy_chunks.jsonl")
    np.save(out / "intermediate" / "X.npy", X)
    write_json({**emb.stats, "model": emb.model, "dimensions": emb.dim}, out / "embedding_stats.json")
    log(f"airweave chunks {len(chunks)}  taxonomy chunks {len(tc)}  embeddings {emb.stats}")
    stage("prepare", t0)

    # 2. discovery (code only)
    t0 = time.perf_counter()
    _, runs, runlog = clustering.run_grid(X, cfg, log)
    runlog.to_csv(out / "intermediate" / "clustering_runs.csv", index=False)
    allc, cand = candidate_clusters.candidates(runs, X, meta, cfg)
    parent = candidate_clusters.hierarchy(cand, cfg)
    cv = _terms_vectorizer(); TF = cv.fit_transform(meta.text)
    terms = clustering.top_terms(cv, TF, dict(zip(cand.cluster_id, cand.members)), cfg["semantic_review"]["top_terms"])
    write_jsonl([{"cluster_id": r.cluster_id, "candidate_key": r.candidate_key, "min_cluster_size": r.min_cluster_size,
                  "min_samples": r.min_samples, "seed_recurrence": r.seed_recurrence, "mean_jaccard": r.mean_jaccard,
                  "mean_overlap": r.mean_overlap, "member_count": r.member_count, "documents": r.documents, "groups": r.groups,
                  "largest_group": r.largest_group, "largest_group_share": r.largest_group_share, "cohesion": r.cohesion,
                  "parent_cluster_id": parent.get(r.cluster_id), "top_terms": terms[r.cluster_id],
                  "member_taxonomy_chunk_ids": list(meta.taxonomy_chunk_id.iloc[r.members])} for r in cand.itertuples()],
                out / "candidate_clusters.jsonl")
    log(f"candidate clusters {len(cand)} from {len(runlog)} HDBSCAN runs")
    stage("discovery", t0)
    if cand.empty:
        raise RuntimeError("no stable candidate clusters: the corpus is too small or too homogeneous for the configured screens")

    # 3. LLM cluster review
    t0 = time.perf_counter()
    rv = cluster_review.review_all(cand, X, meta, terms, cfg, out, ledger, log, llm_transport)
    rv.to_csv(out / "semantic_reviews.csv", index=False)
    rv = rv.astype(object).where(rv.notna(), None)
    log(f"review decisions {rv.decision.value_counts().to_dict()}")
    stage("cluster_review", t0)

    # 4. fine taxonomy (from logged responses only)
    bm = BM25.fit(list(meta.text), cfg); B = bm.transform(list(meta.text))

    def sparse_profile(m):
        v = np.asarray(B[m].mean(0)).ravel(); o = np.argsort(-v)[:50]
        return {bm.vocab[i]: round(float(v[i]), 5) for i in o if v[i] > 0}
    built = builder.build(cand, rv, parent, X, meta, terms, sparse_profile)
    cats = built["categories"]
    if not cats:
        raise RuntimeError("the cluster review accepted no categories")
    log(f"fine categories {len(cats)}  merge recommendations {len(built['merge_recommendations'])}")

    # 5. broad families
    t0 = time.perf_counter()
    text = dict(zip(meta.taxonomy_chunk_id, meta.text))
    bf = consolidation.consolidate(cats, cfg, out, ledger, lambda tid: text[tid], llm_transport)
    write_json(bf, out / "broad_families.json")
    stage("consolidation", t0)

    # 6. freeze + calibrate
    t0 = time.perf_counter()
    fams, prim = bf["families"], bf["primary_family"]
    cats = [{**c, "broad_family_id": prim[c["category_id"]], "broad_family": fams[prim[c["category_id"]]]["family_name"]} for c in cats]
    pos = {t: i for i, t in enumerate(meta.taxonomy_chunk_id)}
    groups = [np.array([pos[t] for t in c["member_taxonomy_chunk_ids"]]) for c in cats]
    # category representation for learned acceptance / wave evolution (generic: members, groups, documents only)
    protos, exemplars = [], []
    for c, g in zip(cats, groups):
        cen = np.asarray(c["dense_prototype"], np.float32)
        c.update({"status": MATURE, "is_sink": False,
                  "stats": category_stats(X[g], cen, meta.group.to_numpy(object)[g], meta.document_id.to_numpy(object)[g],
                                          c["parent_category_id"] is not None, False)})
        protos.append(subprotos(X[g], choose_k(X[g]))); exemplars.append(X[g].astype(np.float32))
    admin = (tags.admin_quality == "ADMINISTRATIVE").to_numpy()
    hy = HybridAssigner(cfg, X, B, meta.group.to_numpy(object), admin, [c["category_id"] for c in cats], groups,
                        {c["category_id"]: c["parent_category_id"] for c in cats}, built["reject_groups"])
    cal = hy.calibrate(cfg["run"]["seed"])
    frozen = hy.freeze()
    families = [{"broad_family_id": fid, "name": f["family_name"], "description": f["description"], "confidence": f["confidence"],
                 "category_ids": list(f["member_ids"]), "source_consolidation_call_id": bf["consolidation_call_id"]} for fid, f in fams.items()]
    lean = [{k: v for k, v in c.items() if k not in ("dense_prototype", "member_taxonomy_chunk_ids", "member_airweave_chunk_ids")} for c in cats]
    frozen_at = datetime.now(timezone.utc)
    input_sha = hashlib.sha256("".join(c.chunk_id + "\x1f" + c.text for c in chunks).encode()).hexdigest()
    final = {"schema_version": artifacts.SCHEMA_VERSION, "taxonomy_id": cfg["run"].get("taxonomy_id") or artifacts.DEFAULT_TAXONOMY_ID,
             "domain": cfg["run"].get("domain"),
             "taxonomy_version": cfg["run"].get("taxonomy_version") or f"taxonomy-{frozen_at:%Y%m%d%H%M%S}",
             "created_at": frozen_at.isoformat(), "frozen_at": frozen_at.isoformat(), "classifier_version": __version__,
             "source": {"airweave_chunks": len(chunks), "documents": len({c.document_id for c in chunks}), "taxonomy_chunks": len(tc),
                        "input_sha256": input_sha, "config_hash": cfg.get("_config_hash")},
             "structure": "broad family -> fine semantic category -> taxonomy chunks",
             "embedding": {"model": cfg["embedding"]["model"], "dimensions": cfg["embedding"]["dimensions"]},
             "taxonomy_chunking": cfg["taxonomy_chunking"], "content_form": cfg.get("content_form", {}),
             "broad_families": families, "categories": lean, "calibration": cal,
             "merge_recommendations": built["merge_recommendations"],
             "consolidation_call_id": bf["consolidation_call_id"], "repair_call_id": bf["repair_call_id"]}
    centroids = np.array([c["dense_prototype"] for c in cats], np.float32)
    cat_sparse = sparse_profiles(B, groups, terms=cfg["assignment"]["profile_terms"])
    final = artifacts.save(out, final, frozen, bm, centroids, cat_sparse, protos, exemplars)
    write_json({c["category_id"]: {"member_taxonomy_chunk_ids": c["member_taxonomy_chunk_ids"],
                                   "member_airweave_chunk_ids": c["member_airweave_chunk_ids"]} for c in cats}, out / "category_members.json")
    pd.DataFrame([{"broad_family_id": c["broad_family_id"], "broad_family": c["broad_family"], "category_id": c["category_id"], "name": c["name"],
                   "description": c["description"], "scope": c["scope"], "industry": c["industry"], "parent_category_id": c["parent_category_id"],
                   "member_count": c["member_count"], "document_count": c["document_count"], "group_count": c["group_count"],
                   "top_terms": "; ".join(c["top_terms"][:10]), "source_cluster_id": c["source_cluster_id"],
                   "source_review_call_id": c["source_review_call_id"]} for c in cats]
                 ).sort_values(["broad_family_id", "member_count"], ascending=[True, False]).to_csv(out / "fine_categories.csv", index=False)
    pd.DataFrame(built["merge_recommendations"]).to_csv(out / "merge_recommendations.csv", index=False)
    pv = out / "prompt_versions"; pv.mkdir(exist_ok=True)
    for v in set(_prompt_versions(cfg).values()):
        for f in PROMPTS_DIR.glob(f"{v}.*"):
            shutil.copy2(f, pv / f.name)
    stage("freeze", t0)
    log(f"frozen: {len(cats)} categories / {len(families)} families; calibration {cal}")

    # 7. optional: enrich the build corpus itself (hybrid labels + discovery-cluster membership)
    if classify_input:
        tax = artifacts.load(out)
        res = frozen.assign(X, B, admin, list(meta.taxonomy_chunk_id))
        member_of = {}
        for c in cats:
            for t in c["member_taxonomy_chunk_ids"]:
                member_of.setdefault(t, []).append(c["category_id"])
        children = []
        for r, t in zip(res, tc):
            topics = [{"category_id": k, "source": "HYBRID_RRF", "score": s} for k, s in zip(r["direct"], r["scores"])]
            topics += [{"category_id": k, "source": "DISCOVERY_CLUSTER_MEMBER", "score": None}
                       for k in member_of.get(t["taxonomy_chunk_id"], []) if k not in r["direct"]]
            ks = [x["category_id"] for x in topics]
            topics = [x for x in topics if not any(x["category_id"] in frozen.anc.get(o, set()) for o in ks if o != x["category_id"])]
            status = r["status"] if (r["status"] == "ASSIGNED" or not topics) else "ASSIGNED_BY_MEMBERSHIP"
            children.append({**{k: t[k] for k in ("taxonomy_chunk_id", "chunk_id", "char_start", "char_end", "content_form")},
                             "status": status, "topics": topics})
        n = write_jsonl(aggregate(chunks, children, tax, cfg), out / "build_enriched_chunks.jsonl")
        log(f"build corpus enriched: {n} Airweave chunks -> build_enriched_chunks.jsonl")

    calls = [json.loads(line) for line in open(out / "llm_calls.jsonl")] if (out / "llm_calls.jsonl").exists() else []
    manifest = {"written_at": datetime.now(timezone.utc).isoformat(), "classifier_version": __version__,
                "taxonomy_id": final["taxonomy_id"], "taxonomy_version": final["taxonomy_version"], "config": public(cfg), "config_hash": cfg.get("_config_hash"),
                "input": {"airweave_chunks": len(chunks), "documents": len({c.document_id for c in chunks}), "taxonomy_chunks": len(tc),
                          "input_sha256": input_sha},
                "result": {"candidate_clusters": int(len(cand)), "categories": len(cats), "broad_families": len(families)},
                "models": {"embedding": cfg["embedding"]["model"], "llm": cfg["llm"]["model"]}, "prompt_versions": _prompt_versions(cfg),
                "prompt_file_hashes": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((out / "prompt_versions").glob("*"))},
                "seeds": {"run": cfg["run"]["seed"], "umap_discovery": cfg["clustering"]["seeds"], "umap_novelty": cfg["novelty"]["seeds"]},
                "api": {"llm_calls": len(calls), "llm_calls_paid": sum(1 for c in calls if not c["cache_hit"]),
                        "llm_status": pd.Series([c["status"] for c in calls]).value_counts().to_dict() if calls else {},
                        "embedding": emb.stats, "cost_ledger": ledger.data},
                "timings_s": {**timings, "total": round(time.perf_counter() - t_start, 2)}, "python": platform.python_version()}
    write_json(manifest, out / "run_manifest.json")
    return manifest


def load_taxonomy(path) -> Taxonomy:
    return artifacts.load(path)
