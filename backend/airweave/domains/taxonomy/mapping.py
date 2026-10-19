"""Classifier records ("topics/1") to the chunk fields of contract airweave-neo/1 (4.2, 4.3).

Every build id the classifier returns (``CAT###``, ``BF##``) is translated to a registry id
through the version's ``registry_map.json``: in ``topic_ids``, ``broad_family_ids`` and every
span's ``c``. A category missing from the map is never written: the label is dropped, its piece
counts as unresolved, and the caller logs it. The classifier saw a chunk's own text only, so
piece offsets are shifted by ``chunk_own_start`` into document coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# Chunk-level statuses (contract 4.2)
ASSIGNED = "ASSIGNED"
PARTIALLY_ASSIGNED = "PARTIALLY_ASSIGNED"
UNRESOLVED = "UNRESOLVED"
ADMINISTRATIVE = "ADMINISTRATIVE"
ERROR = "ERROR"
NOT_LABELLED = "NOT_LABELLED"
STATUSES = (ASSIGNED, PARTIALLY_ASSIGNED, UNRESOLVED, ADMINISTRATIVE, ERROR, NOT_LABELLED)

# Piece (taxonomy child) statuses that are not substantive text (package: NON_SUBSTANTIVE_CHILD)
NON_SUBSTANTIVE_PIECE = frozenset({"BYPASS_ADMINISTRATIVE", "REJECT_NONSEMANTIC"})


class RegistryMapError(ValueError):
    """``registry_map.json`` is malformed or belongs to another version."""


@dataclass(frozen=True)
class RegistryMap:
    """Build ids to registry ids for one taxonomy version (contract section 8)."""

    taxonomy_version: str
    categories: Mapping[str, str]
    families: Mapping[str, str]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RegistryMap":
        """Read ``registry_map.json``; refuse anything but string-to-string maps."""
        try:
            version = data["taxonomy_version"]
            categories, families = data["categories"], data["families"]
        except (KeyError, TypeError) as e:
            raise RegistryMapError(f"registry_map.json is missing {e}") from e
        for name, m in (("categories", categories), ("families", families)):
            if not isinstance(m, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in m.items()
            ):
                raise RegistryMapError(f"registry_map.json: {name} must map strings to strings")
            if any(not k.strip() or not v.strip() for k, v in m.items()):
                raise RegistryMapError(f"registry_map.json: {name} has an empty id")
        return cls(str(version), dict(categories), dict(families))


@dataclass
class ChunkLabels:
    """The label fields of one chunk, as written to Vespa."""

    taxonomy_status: str
    topic_ids: Dict[str, int] = field(default_factory=dict)
    broad_family_ids: List[str] = field(default_factory=list)
    taxonomy_spans: str = "[]"
    taxonomy_piece_count: int = 0
    taxonomy_admin_pieces: int = 0
    taxonomy_unresolved_pieces: int = 0
    untranslated: List[str] = field(default_factory=list)  # build ids missing from the map


def administrative_labels() -> ChunkLabels:
    """Labels for a chunk whose own text is empty or whitespace (contract 10)."""
    return ChunkLabels(taxonomy_status=ADMINISTRATIVE)


def map_record(
    record: Mapping[str, Any],
    registry: RegistryMap,
    own_start: int,
    own_end: Optional[int] = None,
) -> ChunkLabels:
    """Translate one classifier record into the chunk's label fields.

    ``own_start`` and ``own_end`` are the chunk's own range in the document text; piece offsets
    are relative to the own text the classifier received.
    """
    pieces = sorted(record.get("taxonomy_children") or [], key=lambda c: c["char_start"])
    untranslated: List[str] = []
    piece_ids: List[List[str]] = []
    for piece in pieces:
        mapped = []
        for cid in piece.get("topic_ids") or []:
            if cid in registry.categories:
                mapped.append(registry.categories[cid])
            elif cid not in untranslated:
                untranslated.append(cid)
        piece_ids.append(mapped)

    topic_ids: Dict[str, int] = {}
    families: set = set()
    for topic in record.get("topics") or []:
        reg = registry.categories.get(topic["category_id"])
        if reg is None:
            continue  # counted above through its pieces
        # Several classifier categories may map to one registry topic: their pieces add up.
        topic_ids[reg] = topic_ids.get(reg, 0) + (
            int(topic.get("supporting_children") or 0) or sum(reg in ids for ids in piece_ids)
        )
        fam = topic.get("broad_family_id")
        if fam in registry.families:
            families.add(registry.families[fam])
        elif fam and fam not in untranslated:
            untranslated.append(fam)

    substantive = [i for i, p in enumerate(pieces) if p["status"] not in NON_SUBSTANTIVE_PIECE]
    unresolved = sum(1 for i in substantive if not piece_ids[i])
    if topic_ids:
        status = PARTIALLY_ASSIGNED if unresolved else ASSIGNED
    else:
        status = UNRESOLVED if substantive else ADMINISTRATIVE

    return ChunkLabels(
        taxonomy_status=status,
        topic_ids=topic_ids,
        broad_family_ids=sorted(families),
        taxonomy_spans=_spans(pieces, piece_ids, set(topic_ids), own_start, own_end),
        taxonomy_piece_count=len(pieces),
        taxonomy_admin_pieces=len(pieces) - len(substantive),
        taxonomy_unresolved_pieces=unresolved,
        untranslated=untranslated,
    )


def _spans(
    pieces: List[Mapping[str, Any]],
    piece_ids: List[List[str]],
    topics: set,
    own_start: int,
    own_end: Optional[int],
) -> str:
    """One span per run of consecutive pieces carrying a category (contract 4.3)."""
    spans = []
    for cat in sorted(topics):
        run: List[int] = []
        for i, ids in enumerate(piece_ids + [[]]):
            if cat in ids:
                run.append(i)
                continue
            if run:
                s = own_start + pieces[run[0]]["char_start"]
                e = own_start + pieces[run[-1]]["char_end"]
                if own_end is not None:
                    e = min(e, own_end)
                if s < e:
                    spans.append({"c": cat, "s": s, "e": e, "p": [run[0], run[-1]]})
                run = []
    spans.sort(key=lambda sp: (sp["s"], sp["c"]))
    return json.dumps(spans, ensure_ascii=False, separators=(",", ":"))
