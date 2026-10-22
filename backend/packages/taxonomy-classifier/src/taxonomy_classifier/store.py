"""Where built taxonomy artifacts live at run time. A taxonomy is DATA supplied to the engine, never code.

LocalTaxonomyStore is the directory implementation: <root>/<taxonomy_id>/<taxonomy_version>/ holds one built taxonomy
(the taxonomy/artifacts.py layout). Any other backend (object storage, database) only needs the same three operations:
load(taxonomy_id, version) / save(build_dir) / list(). The root is deployment configuration, e.g. an application data
directory; nothing in the engine assumes a repository folder.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .taxonomy import artifacts
from .taxonomy.models import Taxonomy

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def _check(name: str, what: str) -> str:
    if not _SAFE.match(name or ""):
        raise ValueError(f"invalid {what} {name!r}: use letters, digits, '.', '_', '+', '-'")
    return name


class LocalTaxonomyStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def path(self, taxonomy_id: str, version: str) -> Path:
        return self.root / _check(taxonomy_id, "taxonomy_id") / _check(version, "taxonomy_version")

    def list(self) -> list[dict]:
        """One entry per stored taxonomy version, oldest first within a taxonomy_id."""
        out = []
        for f in sorted(self.root.glob("*/*/final_taxonomy.json")):
            d = json.load(open(f))
            out.append({"taxonomy_id": d.get("taxonomy_id", f.parent.parent.name), "taxonomy_version": d["taxonomy_version"],
                        "domain": d.get("domain"), "created_at": d.get("created_at", d.get("frozen_at")),
                        "categories": len(d["categories"]), "path": str(f.parent)})
        return sorted(out, key=lambda r: (r["taxonomy_id"], r["created_at"] or "", r["taxonomy_version"]))

    def versions(self, taxonomy_id: str) -> list[str]:
        return [r["taxonomy_version"] for r in self.list() if r["taxonomy_id"] == taxonomy_id]

    def load(self, taxonomy_id: str, version: str | None = None) -> Taxonomy:
        """version=None -> the most recently created version of taxonomy_id."""
        if version is None:
            vs = self.versions(taxonomy_id)
            if not vs:
                raise FileNotFoundError(f"no taxonomy {taxonomy_id!r} in {self.root}")
            version = vs[-1]
        return artifacts.load(self.path(taxonomy_id, version))

    def save(self, build_dir: str | Path) -> Path:
        """Copy a built taxonomy directory into the store under its own taxonomy_id / taxonomy_version (never overwrites)."""
        src = Path(build_dir)
        tax = artifacts.load(src)
        dst = self.path(tax.taxonomy_id, tax.version)
        if src.resolve() == dst.resolve():
            return dst
        if dst.exists():
            raise FileExistsError(f"{dst} already exists; taxonomy versions are immutable")
        shutil.copytree(src, dst)
        return dst
