"""Input / output schemas.

Input (one Airweave retrieval chunk, i.e. the output of Airweave's SemanticChunker):
    {"chunk_id": "...", "document_id": "...", "text": "...", "metadata": {...optional...}}

Output (the same chunk, enriched; the original text is not repeated):
    {"chunk_id", "document_id", "metadata", "taxonomy": {"status", "topics": [...], "broad_families": [...], ...},
     "content_metadata": {...}}
Semantic topics live under "taxonomy"; content-type metadata (table / narrative / ...) lives under "content_metadata"
and is never mixed into topics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

REQUIRED_FIELDS = ("chunk_id", "document_id", "text")

# Airweave-chunk level statuses
ASSIGNED = "ASSIGNED"                       # >= 1 topic and every substantive taxonomy child is assigned
PARTIALLY_ASSIGNED = "PARTIALLY_ASSIGNED"   # >= 1 topic, but some substantive children are unresolved
UNRESOLVED = "UNRESOLVED"                   # substantive text, no topic (goes to the novelty pool)
ADMINISTRATIVE = "ADMINISTRATIVE"           # only administrative / non-semantic text (signatures, indexes, ...)
CHUNK_STATUSES = (ASSIGNED, PARTIALLY_ASSIGNED, UNRESOLVED, ADMINISTRATIVE)

# Taxonomy-child level statuses (produced by the assigner)
CHILD_ASSIGNED = "ASSIGNED"
CHILD_ASSIGNED_BY_MEMBERSHIP = "ASSIGNED_BY_MEMBERSHIP"   # build corpus only: member of a discovery cluster
CHILD_UNRESOLVED = "UNRESOLVED"
CHILD_BYPASS_ADMINISTRATIVE = "BYPASS_ADMINISTRATIVE"
CHILD_REJECT_NONSEMANTIC = "REJECT_NONSEMANTIC"          # closest to a reject (structural/administrative) prototype
NON_SUBSTANTIVE_CHILD = (CHILD_BYPASS_ADMINISTRATIVE, CHILD_REJECT_NONSEMANTIC)


class InputError(ValueError):
    pass


@dataclass
class AirweaveChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "AirweaveChunk":
        missing = [k for k in REQUIRED_FIELDS if not d.get(k) and d.get(k) != ""]
        if missing:
            raise InputError(f"Airweave chunk is missing required field(s) {missing}: {str(d)[:200]}")
        if not isinstance(d["text"], str):
            raise InputError(f"chunk {d['chunk_id']}: text must be a string")
        return cls(str(d["chunk_id"]), str(d["document_id"]), d["text"], dict(d.get("metadata") or {}))

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "document_id": self.document_id, "text": self.text, "metadata": self.metadata}


def parse_chunks(rows: Iterable[dict]) -> list[AirweaveChunk]:
    out, seen = [], set()
    for r in rows:
        c = r if isinstance(r, AirweaveChunk) else AirweaveChunk.from_dict(r)
        if c.chunk_id in seen:
            raise InputError(f"duplicate chunk_id {c.chunk_id}")
        seen.add(c.chunk_id)
        out.append(c)
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n"); n += 1
    return n


def write_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
