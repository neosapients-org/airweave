"""Offline taxonomies and a filesystem "S3" for the A7 tests: no network, no API spend.

Taxonomies are built with the classifier package's own engine, the way its tests build them: a
bag-of-words fake embedder and a deterministic fake reviewer (``FakeLLM``, copied from the
package's ``examples/offline_fakes.py``). They are then published in the layout Meridian writes
(contract airweave-neo/1, section 8): version folder, ``taxonomy.yaml``, ``registry_map.json``,
``manifest.json``, ``active/<collection>.json`` and, for the learned policy, ``models/<id>/``.
Every company and text here is invented.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

DIM = 64

# Two invented tenants with three topics each
CORPORA: Dict[str, Dict[str, List[str]]] = {
    "acme": {
        "loans": ["late loan payments", "credit loss allowance", "loan charge offs",
                  "delinquent small business loans", "credit quality of the loan book",
                  "nonperforming loans", "loan loss reserve", "borrower defaults"],
        "revenue": ["quarterly revenue growth", "subscription revenue", "revenue by region",
                    "net sales increase", "pricing and volume", "recurring revenue",
                    "revenue guidance", "deferred revenue"],
        "hedging": ["foreign exchange hedges", "interest rate swaps", "hedge accounting",
                    "currency forward contracts", "derivative fair value",
                    "cash flow hedge reserve", "notional amount of swaps", "hedge effectiveness"],
    },
    "globex": {
        "recipes": ["whisk the eggs", "preheat the oven", "simmer the sauce", "knead the dough",
                    "fold in the flour", "season with salt", "chop the onions",
                    "bake until golden"],
        "software": ["install the package", "run the command line",
                     "configure the environment variable", "call the api endpoint",
                     "read the error log", "upgrade the dependency", "write a unit test",
                     "deploy the container"],
        "travel": ["book the train ticket", "visit the old town", "pack a light backpack",
                   "check in at the hostel", "walk along the harbour", "rent a bicycle",
                   "catch the ferry", "explore the mountain trail"],
    },
}


def bow_vector(text: str) -> np.ndarray:
    """Sum of per-word pseudo-random vectors: similar words give similar vectors."""
    v = np.zeros(DIM)
    for w in text.lower().split():
        seed = int.from_bytes(hashlib.sha256(w.strip(".,").encode()).digest()[:4], "little")
        v += np.random.default_rng(seed).normal(size=DIM)
    return v.astype(np.float32)


class BowEmbeddings:
    """Fake embeddings transport: list[str] -> (vectors, tokens). Optional delay per call."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return [bow_vector(t) for t in batch], sum(len(t) // 4 for t in batch)


class FakeLLM:
    """Deterministic stand-in for the chat API (from the package's examples/offline_fakes.py)."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, body):
        name = body["response_format"]["json_schema"]["name"]
        user = body["messages"][1]["content"]
        self.calls.append(name)
        if name.startswith("cluster_semantic_review"):
            terms = re.search(r"Most distinctive terms: (.*)", user).group(1).split(", ")
            admin = bool(re.search(r"\) Page \d+ of \d+", user))
            generic = {"first", "then", "again", "the", "of", "and"}
            label = next((t for t in terms if not set(t.split()) & generic), terms[0]).title()
            out = {"decision": "REJECT_ADMINISTRATIVE" if admin else "ACCEPT", "coherent": True,
                   "category_name": label, "description": f"Content about {label.lower()}.",
                   "scope": "NONE" if admin else "GLOBAL", "industry": None, "company": None,
                   "semantic_or_structural": "SEMANTIC", "duplicate_of": None, "confidence": 0.9,
                   "reason": "fake"}
        elif name.startswith("broad_family_consolidation"):
            ids = sorted(set(re.findall(r"^(CAT\d{3}) \|", user, re.M)))
            out = {"families": [{"family_id": "BF01", "family_name": "All Topics",
                                 "description": "Everything.", "confidence": 0.9, "member_ids": ids}],
                   "possible_misfits": [], "possible_duplicates": [], "parent_child_notes": []}
        else:
            raise AssertionError(name)
        return {"model": "fake", "choices": [{"message": {"content": json.dumps(out)},
                                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


def sentence(phrase: str) -> str:
    return f"First {phrase}, then {phrase} again."


def corpus(vocab: Dict[str, List[str]], n_docs: int = 6, chunks_per_doc: int = 4, seed: int = 0,
           prefix: str = "G") -> List[dict]:
    """Build-corpus chunks: each topic's documents repeat its phrases; one footer per document."""
    rng = np.random.default_rng(seed)
    out = []
    for topic, phrases in vocab.items():
        for d in range(n_docs):
            doc = f"{prefix}_{topic}_{d}"
            for i in range(chunks_per_doc):
                text = " ".join(sentence(p) for p in rng.choice(phrases, size=6))
                out.append({"chunk_id": f"{doc}_{i}", "document_id": doc, "text": text,
                            "metadata": {"site": f"site{d % 4}"}})
            out.append({"chunk_id": f"{doc}_footer", "document_id": doc, "text": f"Page {d} of 9",
                        "metadata": {"site": f"site{d % 4}"}})
    return out


def package_root() -> Path:
    """The classifier package's folder in this repository (its config/ is not installed)."""
    return Path(__file__).resolve().parents[4] / "packages" / "taxonomy-classifier"


def build_config(tmp: Path, description: str) -> dict:
    """The package's example config with the offline test overrides of its own tests."""
    c = copy.deepcopy(yaml.safe_load((package_root() / "config" / "taxonomy.example.yaml").read_text()))
    c["input"]["fields"] = {"group": "site"}
    c["content_form"]["patterns"] = {}
    c["embedding"].update({"model": "fake-bow", "dimensions": DIM,
                           "cache_path": str(tmp / "build-emb.sqlite")})
    c["llm"].update({"cache_dir": str(tmp / "llm_cache"), "corpus_description": description})
    c["clustering"]["hdbscan_settings"] = [[10, 5], [15, 5]]
    c["clustering"]["umap"]["n_neighbors"] = 10
    c["clustering"]["screens"].update({"min_documents": 3})
    c["assignment"]["bm25"]["min_df"] = 2
    return c


def build_taxonomy(tmp: Path, tenant: str, version: str) -> tuple[Path, dict]:
    """Build a taxonomy for one invented tenant; returns (artifact directory, build config)."""
    from taxonomy_classifier import LocalTaxonomyStore, TaxonomyEngine

    cfg = build_config(tmp / tenant, f"{tenant} test documents")
    store = LocalTaxonomyStore(tmp / tenant / "store")
    engine = TaxonomyEngine(cfg, store=store, embed_transport=BowEmbeddings(), llm_transport=FakeLLM())
    tax = engine.discover(corpus(CORPORA[tenant], prefix=tenant.upper()), taxonomy_id=tenant,
                          domain="test", version=version, log=lambda m: None)
    return Path(tax.path), cfg


def write_model(directory: Path, p: float, schema: Optional[str] = None,
                names: Optional[Iterable[str]] = None, model_id: Optional[str] = None) -> tuple[Path, Path]:
    """A constant-probability acceptance model and its manifest (the package's test pattern)."""
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from taxonomy_classifier.assignment import learned as L

    directory.mkdir(parents=True, exist_ok=True)
    y = np.array([1] * int(round(p * 100)) + [0] * (100 - int(round(p * 100))))
    model = DummyClassifier(strategy="prior").fit(
        pd.DataFrame(np.zeros((100, len(L.FEATURE_NAMES))), columns=list(L.FEATURE_NAMES)), y)
    f = directory / "frozen_model.pkl"
    pickle.dump({"model": model, "features": list(L.FEATURE_NAMES)}, open(f, "wb"))
    man = {"model_id": model_id or f"const_{p}", "feature_schema_version": schema or L.FEATURE_SCHEMA_VERSION,
           "feature_names": list(names or L.FEATURE_NAMES), "model_type": "dummy",
           "model_file": f.name, "model_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
           "threshold": 0.80, "max_labels_per_child": 1}
    mp = directory / "manifest.json"
    mp.write_text(json.dumps(man))
    return mp, f


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "x"


@dataclass
class Published:
    tenant: str
    version: str
    manifest_sha256: str
    registry: dict


def publish(s3_root: Path, built: Path, cfg: dict, tenant: str, version: str,
            readable_ids: Iterable[str] = (), policy: str = "calibrated",
            model_dir: Optional[Path] = None, model_id: Optional[str] = None,
            drop_categories: Iterable[str] = ()) -> Published:
    """Publish a built taxonomy into ``<s3_root>/taxonomy/`` as Meridian does (contract 8)."""
    final = json.loads((built / "final_taxonomy.json").read_text())
    prefix = f"taxonomy/tenants/{tenant}/versions/{version}/"
    folder = s3_root / prefix
    folder.mkdir(parents=True, exist_ok=True)
    names = [n for n in ("final_taxonomy.json", "dense_prototypes.npy", "sparse_profiles.npz",
                         "bm25_vocabulary.json", "category_centroids.npy",
                         "category_sparse_profiles.npz", "category_prototypes.npz",
                         "category_exemplars.npz") if (built / n).exists()]
    for n in names:
        shutil.copy(built / n, folder / n)
    tcfg = copy.deepcopy(cfg)
    tcfg["assignment"]["acceptance"] = {"policy": policy, "model_manifest": None, "model_path": None,
                                        "probation_gate": {"p_min": 0.95, "bm25_rank_max": 2, "dense_rank_max": 2}}
    (folder / "taxonomy.yaml").write_text(yaml.safe_dump(tcfg))
    fam_names = {f["broad_family_id"]: f["name"] for f in final.get("broad_families") or []}
    families = {}
    categories = {}
    for c in final["categories"]:
        fam = c.get("broad_family_id") or "BF00"
        fam_reg = f"{tenant}:{slug(fam_names.get(fam, fam))}"
        families[fam] = fam_reg
        if c["category_id"] not in set(drop_categories):
            categories[c["category_id"]] = f"{fam_reg}.{slug(c['name'])}"
    registry = {"taxonomy_version": version, "categories": categories, "families": families}
    (folder / "registry_map.json").write_text(json.dumps(registry, indent=2))
    files = names + ["taxonomy.yaml", "registry_map.json"]
    manifest = {"taxonomy_version": version,
                "files": [{"file": n, "sha256": hashlib.sha256((folder / n).read_bytes()).hexdigest()}
                          for n in files],
                "classifier_version": "1.0.0", "embedding_model": cfg["embedding"]["model"],
                "created_at": "2026-10-15T00:00:00Z", "acceptance_policy": policy}
    if policy == "learned":
        mid = model_id or "acme_test_model"
        manifest["acceptance_model_id"] = mid
        if model_dir is not None:
            pkl = model_dir / "frozen_model.pkl"
            manifest["acceptance_model_sha256"] = hashlib.sha256(pkl.read_bytes()).hexdigest()
            dest = s3_root / "taxonomy" / "models" / mid
            dest.mkdir(parents=True, exist_ok=True)
            for f in ("manifest.json", "frozen_model.pkl"):
                shutil.copy(model_dir / f, dest / f)
    raw = json.dumps(manifest, indent=2).encode()
    (folder / "manifest.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    for rid in readable_ids:
        activate(s3_root, rid, tenant, version, sha)
    return Published(tenant, version, sha, registry)


def activate(s3_root: Path, readable_id: str, tenant: str, version: str, manifest_sha: str,
             generation: int = 1) -> None:
    """Write a collection's active-version file (contract 8)."""
    active = s3_root / "taxonomy" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / f"{readable_id}.json").write_text(json.dumps({
        "collection_readable_id": readable_id, "tenant_id": tenant, "taxonomy_version": version,
        "version_prefix": f"taxonomy/tenants/{tenant}/versions/{version}/",
        "manifest_sha256": manifest_sha, "generation": generation,
        "updated_at": "2026-10-15T00:00:00Z"}))
