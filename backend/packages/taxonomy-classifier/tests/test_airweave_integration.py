"""End-to-end with fake transports: synthetic build -> frozen artifacts -> classify new chunks (0 LLM calls)
-> novelty maintenance -> CLI smoke tests. No network, no API spend."""

import json
import subprocess
import sys

import pytest

from taxonomy_classifier.integration.airweave import TaxonomyClassifier, classify_chunks
from taxonomy_classifier.llm import client as llm_client
from taxonomy_classifier.novelty.maintenance import run_maintenance
from taxonomy_classifier.pipeline import build_taxonomy
from taxonomy_classifier.schemas import CHUNK_STATUSES
from taxonomy_classifier.taxonomy import artifacts

from .conftest import ROOT, FakeEmbeddings, FakeLLM, synthetic_chunks, write_jsonl

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import yaml
    tmp = tmp_path_factory.mktemp("build")
    from .conftest import DIM
    cfg = yaml.safe_load(open(ROOT / "config" / "taxonomy.example.yaml"))
    cfg["embedding"].update({"model": "fake-embedding", "dimensions": DIM, "cache_path": str(tmp / "emb.sqlite")})
    cfg["llm"].update({"cache_dir": str(tmp / "llm_cache"), "corpus_description": "synthetic test documents", "max_cost_usd": 1.0})
    cfg["clustering"]["hdbscan_settings"] = [[10, 5], [15, 5]]
    cfg["clustering"]["umap"]["n_neighbors"] = 10
    cfg["assignment"]["bm25"]["min_df"] = 2
    cfg["run"]["taxonomy_version"] = "test"
    emb, llm = FakeEmbeddings(), FakeLLM()
    out = tmp / "taxonomy"
    man = build_taxonomy(synthetic_chunks(), cfg, out, log=lambda m: None, embed_transport=emb, llm_transport=llm)
    return {"cfg": cfg, "out": out, "manifest": man, "llm": llm, "emb": emb, "tmp": tmp}


def test_build_produces_taxonomy_and_artifacts(built):
    out, man = built["out"], built["manifest"]
    for f in artifacts.REQUIRED + ("fine_categories.csv", "broad_families.json", "run_manifest.json", "category_centroids.npy",
                                   "category_sparse_profiles.npz", "llm_calls.jsonl", "build_enriched_chunks.jsonl"):
        assert (out / f).exists(), f
    assert man["result"]["categories"] == 3 and man["result"]["broad_families"] == 1
    assert any((out / "prompt_versions").glob("cluster_semantic_review_v2.*"))
    tax = artifacts.load(out)
    assert {c["broad_family_id"] for c in tax.categories} == {"BF01"}
    calls = [json.loads(line) for line in open(out / "llm_calls.jsonl")]
    assert {c["stage"] for c in calls} == {"cluster_semantic_review", "broad_family_consolidation"}   # every decision logged
    assert all(c["status"] == "OK" for c in calls)


def test_classify_new_chunks_makes_no_llm_calls(built, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("classification must not call the LLM")
    monkeypatch.setattr(llm_client.LLMClient, "_post", forbidden)
    monkeypatch.setattr(llm_client.LLMClient, "call", forbidden)
    new = synthetic_chunks(n_groups=2, docs_per_group=1, seed=7, prefix="NEW")
    res = TaxonomyClassifier(built["out"], built["cfg"], FakeEmbeddings()).classify(new)
    assert [e["chunk_id"] for e in res.enriched] == [c["chunk_id"] for c in new]          # one record per input, same order
    statuses = [e["taxonomy"]["status"] for e in res.enriched]
    assert set(statuses) <= set(CHUNK_STATUSES)
    sig = [e for e, c in zip(res.enriched, new) if c["text"].startswith("SIGNATURES")]
    assert sig and all(e["taxonomy"]["status"] == "ADMINISTRATIVE" and not e["taxonomy"]["topics"] for e in sig)
    substantive = [e for e in res.enriched if e["taxonomy"]["status"] != "ADMINISTRATIVE"]
    assigned = [e for e in substantive if e["taxonomy"]["status"] == "ASSIGNED"]
    assert len(assigned) >= len(substantive) / 2          # per-category floors are P10 of held-out members: some misses expected
    assert all(len(e["taxonomy"]["topic_ids"]) == 1 for e in assigned)                    # single-topic synthetic chunks
    assert len(res.unresolved_children) == statuses.count("UNRESOLVED")
    e = res.enriched[0]
    assert set(e) == {"chunk_id", "document_id", "metadata", "taxonomy", "content_metadata"}
    assert {"status", "topics", "topic_ids", "broad_family_ids", "taxonomy_chunks", "taxonomy_version"} <= set(e["taxonomy"])


def test_multi_topic_chunk_gets_several_topics(built):
    from .conftest import topic_text
    import numpy as np
    rng = np.random.default_rng(3)
    text = topic_text("tax", rng, 12) + "\n\n" + topic_text("debt", rng, 12)
    out = classify_chunks([{"chunk_id": "AW_MIX", "document_id": "D", "text": text}], built["out"], built["cfg"], FakeEmbeddings())
    tx = out[0]["taxonomy"]
    assert len(tx["taxonomy_chunks"]) >= 2                                 # one Airweave chunk -> several taxonomy children
    child_topics = {t for c in tx["taxonomy_chunks"] for t in c["topic_ids"]}
    assert set(tx["topic_ids"]) == child_topics                             # chunk topics = union of the children's topics
    assert all(t["supporting_taxonomy_chunks"] for t in tx["topics"])


def test_config_mismatch_is_rejected(built):
    import copy
    from taxonomy_classifier.integration.airweave import ConfigMismatch
    bad = copy.deepcopy(built["cfg"]); bad["taxonomy_chunking"]["target_tokens"] = 500
    with pytest.raises(ConfigMismatch):
        TaxonomyClassifier(built["out"], bad, FakeEmbeddings())


def test_novelty_maintenance_cluster_level_calls(built):
    from .conftest import TOPICS
    TOPICS["novel"] = ["warehouse robotics", "autonomous forklifts", "picking automation", "conveyor sensors"]
    try:
        import numpy as np
        from .conftest import BASIS, DIM
        BASIS["novel"] = np.random.default_rng(99).normal(size=DIM)
        rng = np.random.default_rng(5)
        pool = [{"taxonomy_chunk_id": f"N{i}::tc000", "chunk_id": f"N{i}", "document_id": f"DOC{i % 6}", "group": f"G{i % 4}",
                 "text": " ".join(f"The firm expanded {p} across sites in the period." for p in rng.choice(TOPICS["novel"], 6))} for i in range(40)]
        llm = FakeLLM()
        s = run_maintenance(artifacts.load(built["out"]), pool, built["cfg"], built["tmp"] / "novelty", log=lambda m: None,
                            embed_transport=FakeEmbeddings(), llm_transport=llm)
    finally:
        TOPICS.pop("novel"); BASIS.pop("novel", None)
    assert s["pool_size"] == 40
    assert 1 <= len(llm.calls) <= s["residual_clusters"] < 40          # one call per residual cluster, never per chunk
    assert s["new_categories"] and s["new_categories"][0]["name"] == "Novel Topic"
    assert (built["tmp"] / "novelty" / "taxonomy_maintenance_proposal.json").exists()


def test_cli_help():
    for s in ("build_taxonomy.py", "classify_airweave_chunks.py", "run_novelty_maintenance.py"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s), "--help"], capture_output=True, text=True)
        assert r.returncode == 0 and "usage" in r.stdout


def test_cli_classify_smoke(built, tmp_path):
    """Classification CLI on chunks whose embeddings are already cached (no network)."""
    import yaml
    new = synthetic_chunks(n_groups=1, docs_per_group=1, seed=0)          # same texts as the build corpus -> cache hits
    inp = write_jsonl(new, tmp_path / "in.jsonl")
    cfgp = tmp_path / "cfg.yaml"; yaml.safe_dump(built["cfg"], open(cfgp, "w"))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "classify_airweave_chunks.py"), "--taxonomy", str(built["out"] / "final_taxonomy.json"),
                        "--input", str(inp), "--output", str(tmp_path / "out.jsonl"), "--config", str(cfgp),
                        "--unresolved-output", str(tmp_path / "unresolved.jsonl")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout)
    assert summary["enriched_chunks"] == len(new) and summary["llm_calls"] == 0 and summary["embedding"]["api_requests"] == 0
    assert len(open(tmp_path / "out.jsonl").readlines()) == len(new)
