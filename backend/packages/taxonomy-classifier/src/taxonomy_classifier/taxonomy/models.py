"""In-memory view of a frozen taxonomy (everything normal classification needs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..assignment.classifier import FrozenAssigner
from ..assignment.sparse import BM25


@dataclass
class Taxonomy:
    path: Path                                   # directory holding final_taxonomy.json and the other artifacts
    final: dict                                  # parsed final_taxonomy.json
    assigner: FrozenAssigner                     # dense prototypes + sparse profiles + calibrated thresholds
    bm25: BM25                                   # frozen vocabulary / IDF
    category_centroids: np.ndarray | None = None  # all categories (novelty maintenance only)
    category_sparse: object | None = None          # all categories (novelty maintenance only)
    _info: dict = field(default_factory=dict, repr=False)
    _by: dict = field(default_factory=dict, repr=False)

    @property
    def version(self) -> str:
        return self.final["taxonomy_version"]

    @property
    def taxonomy_id(self) -> str:
        return self.final.get("taxonomy_id", "default")

    @property
    def domain(self) -> str | None:
        """Free-form label (e.g. "legal_contracts"); metadata only, never used for control flow."""
        return self.final.get("domain")

    @property
    def schema_version(self) -> str:
        return self.final.get("schema_version", "legacy")

    @property
    def categories(self) -> list[dict]:
        return self.final["categories"]

    @property
    def broad_families(self) -> list[dict]:
        return self.final["broad_families"]

    def parent(self, category_id: str) -> str | None:
        return self._by_id()[category_id].get("parent_category_id")

    def ancestors(self, category_id: str) -> list[str]:
        """Immediate parent first, then outward to the root. Ancestors are metadata: they are not assigned labels."""
        out, p, seen = [], self.parent(category_id), {category_id}
        while p and p not in seen:
            seen.add(p); out.append(p); p = self._by_id().get(p, {}).get("parent_category_id")
        return out

    def children(self, category_id: str) -> list[str]:
        return [c["category_id"] for c in self.categories if c.get("parent_category_id") == category_id]

    def _by_id(self) -> dict:
        if not self._by:
            self._by = {c["category_id"]: c for c in self.categories}
        return self._by

    def info(self, category_id: str) -> dict:
        """category_id -> {name, broad_family_id, broad_family_name} (also covers categories added by novelty proposals)."""
        if not self._info:
            self._info = {c["category_id"]: {"name": c["name"], "broad_family_id": c.get("broad_family_id"),
                                             "broad_family_name": c.get("broad_family")} for c in self.categories}
        return self._info.get(category_id, {"name": category_id, "broad_family_id": None, "broad_family_name": None})
