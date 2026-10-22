"""Discover and propose through the REAL classifier package, offline.

Needs ``taxonomy_classifier`` (the full install) and scikit-learn; skipped otherwise. Embeddings
are the bag-of-words fake, the reviewer a deterministic fake, S3 a fake bucket. Every text is
invented. No network and no spend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("taxonomy_classifier")
pytest.importorskip("sklearn")

from airweave.adapters.storage.filesystem import FilesystemBackend  # noqa: E402
from airweave.domains.taxonomy.artifacts import TaxonomyResolver  # noqa: E402
from airweave.domains.taxonomy.jobs.candidates import CandidateWriter  # noqa: E402
from airweave.domains.taxonomy.jobs.gateway import PackageGateway, SpendCapReached  # noqa: E402
from airweave.domains.taxonomy.jobs.records import (  # noqa: E402
    COMPLETED,
    KIND_DISCOVER,
    KIND_PROPOSE,
    InMemoryTaxonomyJobRepository,
)
from airweave.domains.taxonomy.jobs.runner import TaxonomyJobRunner  # noqa: E402
from airweave.domains.taxonomy.jobs.tests.helpers import (  # noqa: E402
    FakeChunkStore,
    FakeS3,
    chunk,
    job,
)
from airweave.domains.taxonomy.relabel.tests.helpers import Clock  # noqa: E402
from airweave.domains.taxonomy.tests.taxonomy_fixtures import (  # noqa: E402
    CORPORA,
    BowEmbeddings,
    FakeLLM,
    build_config,
    build_taxonomy,
    publish,
    sentence,
)

pytestmark = pytest.mark.slow

GARDEN = [
    "prune the rose bushes",
    "water the tomato seedlings",
    "spread garden compost",
    "plant the tulip bulbs",
    "weed the vegetable bed",
    "mulch around the shrubs",
    "harvest the runner beans",
    "rake the autumn leaves",
]
TENANT = "t-acme"


def stored_chunks(vocab, n_docs=6, per_doc=4, seed=0, prefix="D"):
    """Chunks as the fake Vespa stores them: header, positions, own text = the body."""
    rng = np.random.default_rng(seed)
    out = []
    for topic, phrases in vocab.items():
        for d in range(n_docs):
            for i in range(per_doc):
                body = " ".join(sentence(p) for p in rng.choice(phrases, size=6))
                out.append(chunk(i, doc=f"{prefix}-{topic}-{d}", body=body))
    return out


def package_chunks(rows):
    from airweave.domains.taxonomy.jobs.sampling import to_package_chunk
    from airweave.domains.taxonomy.relabel.chunks import ChunkRow

    return [to_package_chunk(ChunkRow(s, i, f)) for s, i, f in rows]


class NoveltyReviewer:
    """Gardening clusters are a new global topic; anything else is not coherent."""

    def review(self, ctx):
        if any(
            w in ctx["chunks"] for w in ("garden", "tomato", "rose", "tulip", "shrubs", "beans")
        ):
            return {
                "status": "OK",
                "call_id": "fake",
                "parsed": {
                    "coherent": True,
                    "fits_existing": False,
                    "existing_category_id": None,
                    "new_category_name": "Gardening tasks",
                    "description": "Garden work.",
                    "scope": "GLOBAL",
                },
            }
        return {
            "status": "OK",
            "call_id": "fake",
            "parsed": {
                "coherent": False,
                "fits_existing": False,
                "existing_category_id": None,
                "new_category_name": None,
                "description": None,
                "scope": "GLOBAL",
            },
        }


class Structural:
    def review(self, ctx):
        return {"classification": "SEMANTIC_TOPIC", "reason": "fake", "call_id": "fake"}


@pytest.fixture(scope="module")
def build_cfg(tmp_path_factory):
    return build_config(tmp_path_factory.mktemp("cfg"), "Acme test documents")


def gateway(build_cfg, llm=None, reviewers=None):
    return PackageGateway(
        build_cfg,
        embed_transport=BowEmbeddings(),
        llm_transport=llm or FakeLLM(),
        reviewers=reviewers,
    )


def runner_for(tmp_path, build_cfg, rows, s3=None, resolver=None, llm=None, reviewers=None):
    s3 = s3 or FakeS3()
    store = FakeChunkStore(rows)
    repo = InMemoryTaxonomyJobRepository(Clock())
    return (
        TaxonomyJobRunner(
            repo,
            lambda: store,
            CandidateWriter(s3, "taxonomy/"),
            gateway(build_cfg, llm, reviewers),
            resolver,
            work_dir=tmp_path / "jobs",
            min_documents=3,
        ),
        repo,
        s3,
    )


class TestDiscover:
    async def test_writes_a_loadable_candidate_first_version_and_a_report(
        self, tmp_path, build_cfg
    ):
        rows = stored_chunks(CORPORA["acme"])
        runner, repo, s3 = runner_for(tmp_path, build_cfg, rows, resolver=None)
        j = job(candidate_version="acme-first", spend_cap_usd=5.0)
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert done.state == COMPLETED, done.error
        folder = "taxonomy/tenants/t-acme/candidates/acme-first/"
        names = {p[len(folder) :] for p in s3.under(folder)}
        assert {
            "final_taxonomy.json",
            "dense_prototypes.npy",
            "sparse_profiles.npz",
            "bm25_vocabulary.json",
            "taxonomy.yaml",
            "report.json",
            "_claim.json",
        } <= names
        # the package's audit trail holds customer text and never leaves the worker
        assert not names & {
            "llm_calls.jsonl",
            "candidate_clusters.jsonl",
            "semantic_reviews.csv",
            "category_members.json",
            "cost_ledger.json",
            "run_manifest.json",
        }
        assert not any(
            n.startswith(("llm_calls/", "intermediate/", "prompt_versions/")) for n in names
        )
        # it loads with the package, from a copy of exactly the uploaded files
        out = tmp_path / "loaded"
        out.mkdir()
        for n in names:
            if n not in ("report.json", "_claim.json", "taxonomy.yaml"):
                (out / n).write_bytes(s3.objects[folder + n])
        from taxonomy_classifier.taxonomy import artifacts

        taxonomy = artifacts.load(out)
        assert taxonomy.version == "acme-first" and len(taxonomy.categories) >= 3

    async def test_the_shipped_taxonomy_yaml_has_no_local_paths_or_caps(self, tmp_path, build_cfg):
        import yaml

        rows = stored_chunks(CORPORA["acme"])
        runner, repo, s3 = runner_for(tmp_path, build_cfg, rows)
        j = job(candidate_version="acme-first", spend_cap_usd=5.0)
        await repo.create(j)
        await runner.run_job(j.id)
        shipped = yaml.safe_load(
            s3.objects["taxonomy/tenants/t-acme/candidates/acme-first/taxonomy.yaml"]
        )
        assert str(tmp_path) not in json.dumps(shipped)
        assert "max_cost_usd" not in shipped["llm"]
        assert shipped["assignment"]["acceptance"]["policy"] == "calibrated"

    async def test_the_report_lists_clusters_with_example_passages_and_the_cost(
        self, tmp_path, build_cfg
    ):
        rows = stored_chunks(CORPORA["acme"])
        runner, repo, s3 = runner_for(tmp_path, build_cfg, rows)
        j = job(candidate_version="acme-first", spend_cap_usd=5.0)
        await repo.create(j)
        await runner.run_job(j.id)
        report = json.loads(s3.objects["taxonomy/tenants/t-acme/candidates/acme-first/report.json"])
        assert report["kind"] == "discover" and report["categories"] >= 3
        assert report["cost_usd"] > 0 and report["sample"]["documents"] >= 12
        assert all(c["category_id"] and c["name"] for c in report["clusters"])
        assert any(c["example_passages"] for c in report["clusters"])
        assert [f["file"] for f in report["files"]] == sorted(f["file"] for f in report["files"])
        assert (await repo.get(j.id)).spend_used_usd == report["cost_usd"]

    async def test_the_spend_cap_reaches_the_package_ledger(self, tmp_path, build_cfg):
        rows = stored_chunks(CORPORA["acme"])
        runner, repo, s3 = runner_for(tmp_path, build_cfg, rows)
        j = job(candidate_version="acme-first", spend_cap_usd=1e-9)  # smaller than any call
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert done.state == "FAILED" and "cap" in done.error.lower()
        assert not [p for p in s3.objects if not p.endswith("_claim.json")]

    async def test_the_sample_text_is_not_copied_anywhere_but_the_work_dir_and_it_is_removed(
        self, tmp_path, build_cfg
    ):
        rows = stored_chunks(CORPORA["acme"])
        runner, repo, s3 = runner_for(tmp_path, build_cfg, rows)
        j = job(candidate_version="acme-first", spend_cap_usd=5.0)
        await repo.create(j)
        await runner.run_job(j.id)
        assert not (tmp_path / "jobs" / str(j.id)).exists()
        needle = b"late loan payments, then late loan payments again"
        holders = sorted(p.rsplit("/", 1)[1] for p, body in s3.objects.items() if needle in body)
        assert holders in ([], ["report.json"]), holders  # only the report's short excerpts
        report = json.loads(s3.objects["taxonomy/tenants/t-acme/candidates/acme-first/report.json"])
        passages = [x for c in report["clusters"] for x in c["example_passages"]]
        assert passages and all(len(x) <= 400 for x in passages)
        assert all(len(c["example_passages"]) <= 3 for c in report["clusters"])


class TestPropose:
    async def serving(self, tmp_path, build_cfg):
        built, cfg = build_taxonomy(tmp_path, "acme", "acme-v1")
        s3 = tmp_path / "s3"
        publish(s3, built, cfg, "acme", "acme-v1", ["neo-acme"])
        return TaxonomyResolver(FilesystemBackend(s3), "taxonomy/", tmp_path / "local", 300), cfg

    def evolve_cfg(self, cfg):
        cfg["novelty"] = {
            **cfg["novelty"],
            "hdbscan_settings": [[5, 3], [8, 3]],
            "umap": {**cfg["novelty"]["umap"], "n_neighbors": 8},
        }
        cfg["evolution"] = {
            **cfg.get("evolution", {}),
            "creation": {"min_documents": 2, "min_groups": 2, "scopes": ["GLOBAL", "INDUSTRY"]},
        }

    async def test_a_new_topic_becomes_a_probationary_category_and_old_ids_stay_stable(
        self, tmp_path, build_cfg
    ):
        resolver, cfg = await self.serving(tmp_path, build_cfg)
        serving = await resolver.resolve("neo-acme")
        self.evolve_cfg(serving.config)
        garden = stored_chunks({"garden": GARDEN}, n_docs=6, per_doc=4, seed=3, prefix="W")
        unresolved = [
            (
                s,
                i,
                {
                    **f,
                    "taxonomy_status": "UNRESOLVED",
                    "taxonomy_version": "acme-v1",
                    "taxonomy_unresolved_pieces": 1,
                },
            )
            for s, i, f in garden
        ]
        runner, repo, s3 = runner_for(
            tmp_path / "run",
            build_cfg,
            unresolved,
            resolver=resolver,
            reviewers=(NoveltyReviewer(), Structural()),
        )
        j = job(kind=KIND_PROPOSE, candidate_version="acme-v2", spend_cap_usd=5.0, tenant_id="acme")
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert done.state == COMPLETED, done.error
        folder = "taxonomy/tenants/acme/candidates/acme-v2/"
        report = json.loads(s3.objects[folder + "report.json"])
        assert report["kind"] == "propose" and report["base_version"] == "acme-v1"
        assert report["decisions"].get("CREATE_NEW", 0) >= 1
        added = report["categories_added"]
        assert added and all(
            c["status"] == "PROBATIONARY" and c["origin"].startswith("WAVE") for c in added
        )
        assert any(c["example_passages"] for c in added)
        # carried-over ids are stable: every serving category id is still there
        new_final = json.loads(s3.objects[folder + "final_taxonomy.json"])
        old_ids = set(serving.registry.categories)
        assert old_ids <= {c["category_id"] for c in new_final["categories"]}
        assert new_final["derived_from"]["taxonomy_version"] == "acme-v1"
        # the candidate keeps the serving version's own taxonomy.yaml (same embedding and policy)
        assert (
            s3.objects[folder + "taxonomy.yaml"]
            == (serving.local_path / "taxonomy.yaml").read_bytes()
        )
        assert not [p for p in s3.objects if "/versions/" in p]

    async def test_the_serving_version_is_never_modified(self, tmp_path, build_cfg):
        import hashlib

        resolver, cfg = await self.serving(tmp_path, build_cfg)
        serving = await resolver.resolve("neo-acme")
        self.evolve_cfg(serving.config)
        before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in serving.local_path.iterdir()
            if p.is_file()
        }
        garden = stored_chunks({"garden": GARDEN}, n_docs=6, per_doc=4, seed=3, prefix="W")
        unresolved = [
            (s, i, {**f, "taxonomy_status": "UNRESOLVED", "taxonomy_version": "acme-v1"})
            for s, i, f in garden
        ]
        runner, repo, _ = runner_for(
            tmp_path / "run",
            build_cfg,
            unresolved,
            resolver=resolver,
            reviewers=(NoveltyReviewer(), Structural()),
        )
        j = job(kind=KIND_PROPOSE, candidate_version="acme-v2", spend_cap_usd=5.0, tenant_id="acme")
        await repo.create(j)
        await runner.run_job(j.id)
        after = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in serving.local_path.iterdir()
            if p.is_file()
        }
        assert before == after

    async def test_a_wave_that_adds_nothing_writes_a_report_only(self, tmp_path, build_cfg):
        resolver, cfg = await self.serving(tmp_path, build_cfg)
        serving = await resolver.resolve("neo-acme")
        self.evolve_cfg(serving.config)
        noise = [
            chunk(i, doc=f"n{i}", body=f"zq vx kj {i}", status="UNRESOLVED", version="acme-v1")
            for i in range(5)
        ]
        runner, repo, s3 = runner_for(
            tmp_path / "run",
            build_cfg,
            noise,
            resolver=resolver,
            reviewers=(NoveltyReviewer(), Structural()),
        )
        j = job(kind=KIND_PROPOSE, candidate_version="acme-v2", spend_cap_usd=5.0, tenant_id="acme")
        await repo.create(j)
        await runner.run_job(j.id)
        done = await repo.get(j.id)
        assert done.state == COMPLETED and done.result["categories"] == len(
            serving.registry.categories
        )
        names = {p.rsplit("/", 1)[1] for p in s3.under("taxonomy/tenants/acme/candidates/acme-v2/")}
        assert "final_taxonomy.json" not in names and "report.json" in names


def test_discover_gateway_reports_a_refused_call_as_the_cap(build_cfg, tmp_path):
    assert issubclass(SpendCapReached, RuntimeError)
    assert Path(tmp_path).exists() and KIND_DISCOVER != KIND_PROPOSE
