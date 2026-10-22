"""Canonical (human-readable, vector-free) taxonomy exports: <root>/registry.json + <root>/<domain>/taxonomy.json. Stable API.

    from taxonomy_classifier.canonical import load_registry, load_canonical_taxonomy, validate_taxonomy
    tax = load_canonical_taxonomy("/data/taxonomies/legal_contracts/taxonomy.json")   # any path, any domain
    tax = load_canonical_taxonomy("legal_contracts", root="/data/taxonomies")          # by registry name
    tax.get(cid)["name"]; tax.ancestors(cid); tax.children(cid); tax.descendants(cid)

Domain names are free-form registry keys, never an enum. Exports describe categories; they do not contain the vectors
needed for assignment (those live with a built taxonomy directory, see taxonomy/artifacts.py).
`category_id` is the stable identifier to store; `name` is display text.

Where the root comes from: the `root` argument, else $TAXONOMY_CLASSIFIER_TAXONOMIES_DIR, else the `taxonomies/` folder
of a development checkout (reference exports from the research phase). Deployments should pass a root or set the variable.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSIONS = ("taxonomy-export/1",)
VALID_STATUSES = ("MATURE", "PROBATIONARY", "ACCEPTED")
REQUIRED_FIELDS = ("category_id", "name", "description", "status", "domain", "parent_id", "ancestor_ids", "children_ids", "depth")
ROOT_ENV = "TAXONOMY_CLASSIFIER_TAXONOMIES_DIR"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "taxonomies"   # development checkout only (reference exports)


def default_root() -> Path:
    env = os.environ.get(ROOT_ENV)
    root = Path(env) if env else DEFAULT_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"no canonical taxonomy root at {root}: pass root=... or set ${ROOT_ENV}")
    return root


@dataclass
class CanonicalTaxonomy:
    domain: str
    data: dict

    def __post_init__(self):
        self._by_id = {c["category_id"]: c for c in self.data["categories"]}

    @property
    def version(self) -> str:
        return self.data["taxonomy_version"]

    @property
    def categories(self) -> list[dict]:
        return self.data["categories"]

    @property
    def families(self) -> list[dict]:
        return self.data["families"]

    def get(self, category_id: str) -> dict:
        return self._by_id[category_id]

    def __contains__(self, category_id: str) -> bool:
        return category_id in self._by_id

    def parent(self, category_id: str) -> str | None:
        return self._by_id[category_id]["parent_id"]

    def ancestors(self, category_id: str) -> list[str]:
        """Immediate parent first, then outward to the root."""
        return list(self._by_id[category_id]["ancestor_ids"])

    def children(self, category_id: str) -> list[str]:
        return list(self._by_id[category_id]["children_ids"])

    def descendants(self, category_id: str) -> list[str]:
        out, stack = [], list(self.children(category_id))
        while stack:
            c = stack.pop(0); out.append(c); stack.extend(self.children(c))
        return out

    def roots(self) -> list[str]:
        return [c["category_id"] for c in self.categories if not c["parent_id"]]


def load_registry(root: str | Path | None = None) -> dict:
    return json.load(open(Path(root or default_root()) / "registry.json"))


def load_canonical_taxonomy(domain_or_path: str | Path, root: str | Path | None = None) -> CanonicalTaxonomy:
    """A path to a taxonomy.json, or a domain name listed in <root>/registry.json."""
    p = Path(domain_or_path)
    if not p.suffix:
        root = Path(root or default_root())
        reg = load_registry(root)
        if str(domain_or_path) not in reg["domains"]:
            raise KeyError(f"unknown domain {domain_or_path!r}; registry has {sorted(reg['domains'])}")
        p = Path(root) / reg["domains"][str(domain_or_path)]["taxonomy_path"]
    data = json.load(open(p))
    return CanonicalTaxonomy(data["domain"], data)


def save_canonical_taxonomy(tax: CanonicalTaxonomy, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(tax.data, indent=1, ensure_ascii=False) + "\n")


def validate_taxonomy(data: dict) -> dict:
    """Returns {"errors": [...], "warnings": [...], "stats": {...}}. Duplicate names are warnings, never merged."""
    errors, warnings = [], []
    if data.get("schema_version") not in SCHEMA_VERSIONS:
        errors.append(f"unknown schema_version {data.get('schema_version')!r}")
    cats = data.get("categories", [])
    ids = [c.get("category_id") for c in cats]
    for cid, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate category_id {cid} ({n}x)")
    by_id = {c["category_id"]: c for c in cats if c.get("category_id")}
    fam_ids = {f["family_id"] for f in data.get("families", [])}
    for c in cats:
        cid = c.get("category_id")
        for f in REQUIRED_FIELDS:
            if f not in c:
                errors.append(f"{cid}: missing field {f}")
        if c.get("status") not in VALID_STATUSES:
            errors.append(f"{cid}: invalid status {c.get('status')!r}")
        if not (c.get("description") or "").strip():
            warnings.append(f"{cid}: missing description")
        if c.get("broad_family_id") and fam_ids and c["broad_family_id"] not in fam_ids:
            errors.append(f"{cid}: unknown family {c['broad_family_id']}")
        p = c.get("parent_id")
        if p and p not in by_id:
            errors.append(f"{cid}: parent {p} does not exist")
        for k in c.get("children_ids", []):
            if k not in by_id:
                errors.append(f"{cid}: child {k} does not exist")
            elif by_id[k].get("parent_id") != cid:
                errors.append(f"{cid}: child {k} does not point back to it")
        # cycle check + consistency of the stored ancestors / depth
        anc, q, seen = [], p, {cid}
        while q:
            if q in seen:
                errors.append(f"{cid}: cycle through {q}"); break
            seen.add(q); anc.append(q); q = by_id.get(q, {}).get("parent_id")
        else:
            if c.get("ancestor_ids") != anc:
                errors.append(f"{cid}: ancestor_ids {c.get('ancestor_ids')} != parent chain {anc}")
            if c.get("depth") != len(anc) + 1:
                errors.append(f"{cid}: depth {c.get('depth')} != {len(anc) + 1}")
        if p and p in by_id and c.get("broad_family_id") and by_id[p].get("broad_family_id") not in (None, c["broad_family_id"]):
            warnings.append(f"{cid}: parent {p} is in another family ({by_id[p]['broad_family_id']} vs {c['broad_family_id']})")
    names = Counter((c.get("name") or "").strip().lower() for c in cats)
    for n, k in names.items():
        if k > 1:
            dup = [c["category_id"] for c in cats if (c.get("name") or "").strip().lower() == n]
            warnings.append(f"duplicate name {n!r}: {dup}")
    stats = {"categories": len(cats), "families": len(data.get("families", [])), "with_parent": sum(1 for c in cats if c.get("parent_id")),
             "max_depth": max((c.get("depth", 0) for c in cats), default=0), "statuses": dict(Counter(c.get("status") for c in cats))}
    return {"errors": errors, "warnings": warnings, "stats": stats}
