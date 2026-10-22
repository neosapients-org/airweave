"""Save / load the frozen taxonomy artifacts.

REQUIRED for normal classification (all written by the build into one directory):
  final_taxonomy.json        broad families -> fine categories, calibration thresholds, artifact checksums
  dense_prototypes.npy       assignment prototypes (positive categories, then reject prototypes)
  sparse_profiles.npz        BM25 profiles for the same rows
  bm25_vocabulary.json       frozen BM25 vocabulary + IDF
  (+ the config file used for classification: chunking + embedding settings must match the build)
Used by novelty maintenance, learned acceptance and wave evolution (every category, rows aligned with final_taxonomy.json):
  category_centroids.npy, category_sparse_profiles.npz, category_prototypes.npz, category_exemplars.npz
Identity (final_taxonomy.json): schema_version, taxonomy_id, domain (free-form metadata), taxonomy_version, created_at,
source (build input summary). The engine never branches on taxonomy_id or domain.
Evidence / inspection only (not needed at run time):
  fine_categories.csv, broad_families.json, category_members.json, semantic_reviews.csv, candidate_clusters.jsonl,
  merge_recommendations.csv, llm_calls/, llm_calls.jsonl, prompt_versions/, run_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..assignment.classifier import FrozenAssigner
from ..assignment.sparse import BM25
from ..schemas import write_json
from . import representation
from .models import Taxonomy

REQUIRED = ("final_taxonomy.json", "dense_prototypes.npy", "sparse_profiles.npz", "bm25_vocabulary.json")
SCHEMA_VERSION = "taxonomy-artifact/1"   # final_taxonomy.json format; artifacts written before it was recorded load as legacy
KNOWN_SCHEMA_VERSIONS = (SCHEMA_VERSION,)
DEFAULT_TAXONOMY_ID = "default"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(out: Path, final: dict, assigner: FrozenAssigner, bm25: BM25, centroids: np.ndarray, cat_sparse,
         prototypes: list | None = None, exemplars: list | None = None) -> dict:
    """prototypes / exemplars: per category (final["categories"] order) arrays of shape (k, d); needed by learned acceptance."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dense_prototypes.npy", np.asarray(assigner.Dp, np.float32))
    sp.save_npz(out / "sparse_profiles.npz", sp.csr_matrix(assigner.Sp))
    write_json(bm25.state(), out / "bm25_vocabulary.json")
    np.save(out / "category_centroids.npy", np.asarray(centroids, np.float32))
    sp.save_npz(out / "category_sparse_profiles.npz", sp.csr_matrix(cat_sparse))
    files = ["dense_prototypes.npy", "sparse_profiles.npz", "bm25_vocabulary.json", "category_centroids.npy", "category_sparse_profiles.npz"]
    dim = np.asarray(centroids).shape[1]
    for name, arrs in (("category_prototypes.npz", prototypes), ("category_exemplars.npz", exemplars)):
        if arrs is not None:
            V, off = representation.stack(arrs, dim)
            np.savez(out / name, vectors=V, offsets=off); files.append(name)
    final = {**final, "assignment": {"prototype_category_ids": assigner.pos_ids, "n_reject_prototypes": assigner.n_reject,
                                     "category_floors": dict(zip(assigner.pos_ids, map(float, assigner.fl))),
                                     "tau_rrf": assigner.tau, "sibling_delta": assigner.delta, "window": assigner.window,
                                     "rrf_k": assigner.rrf_k, "parent": assigner.parent},
             "artifacts": {f: _sha(out / f) for f in files}}
    write_json(final, out / "final_taxonomy.json")
    return final


def load(path: str | Path, verify: bool = True) -> Taxonomy:
    """path: the taxonomy directory or its final_taxonomy.json."""
    p = Path(path)
    d = p.parent if p.is_file() else p
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"taxonomy directory {d} is missing required artifact(s): {missing}")
    final = json.load(open(d / "final_taxonomy.json"))
    sv = final.get("schema_version")
    if sv is not None and sv not in KNOWN_SCHEMA_VERSIONS:
        raise ValueError(f"{d / 'final_taxonomy.json'}: unsupported schema_version {sv!r} (this engine reads {KNOWN_SCHEMA_VERSIONS})")
    if verify:
        for f, h in final.get("artifacts", {}).items():   # an artifact the manifest lists must be there and match
            if not (d / f).exists():
                raise FileNotFoundError(f"artifact {f} is listed in final_taxonomy.json but missing from {d}")
            if _sha(d / f) != h:
                raise ValueError(f"artifact {f} does not match the checksum recorded in final_taxonomy.json")
    a = final["assignment"]
    pos = a["prototype_category_ids"]
    assigner = FrozenAssigner(pos, a["parent"], np.load(d / "dense_prototypes.npy"), sp.load_npz(d / "sparse_profiles.npz").tocsr(),
                              a["n_reject_prototypes"], [a["category_floors"][c] for c in pos], a["tau_rrf"], a["sibling_delta"],
                              a["window"], a["rrf_k"])
    bm25 = BM25.from_state(json.load(open(d / "bm25_vocabulary.json")))
    cen = np.load(d / "category_centroids.npy") if (d / "category_centroids.npy").exists() else None
    cs = sp.load_npz(d / "category_sparse_profiles.npz").tocsr() if (d / "category_sparse_profiles.npz").exists() else None
    return Taxonomy(d, final, assigner, bm25, cen, cs)


def load_representation(tax: Taxonomy) -> representation.CategoryRepresentation:
    """Prototypes, exemplars, statistics and statuses of every category (learned acceptance / wave evolution)."""
    d = tax.path
    missing = [f for f in ("category_prototypes.npz", "category_exemplars.npz", "category_centroids.npy", "category_sparse_profiles.npz")
               if not (d / f).exists()]
    if missing or any("stats" not in c for c in tax.categories):
        raise FileNotFoundError(f"taxonomy {tax.taxonomy_id}/{tax.version} has no category representation for learned acceptance "
                                f"(missing {missing or 'category stats'}); rebuild it with this engine version")
    pz, ez = np.load(d / "category_prototypes.npz"), np.load(d / "category_exemplars.npz")
    a = tax.final["assignment"]
    reject = tax.assigner.Dp[len(a["prototype_category_ids"]):]
    return representation.from_taxonomy(tax.final, tax.category_centroids, tax.category_sparse, representation.unstack(pz["vectors"], pz["offsets"]),
                                        representation.unstack(ez["vectors"], ez["offsets"]), reject)
