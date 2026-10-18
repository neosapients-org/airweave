"""Chunk statistics for a collection against a target version (platform port ``chunkStats``).

Computed by one scroll over the collection's chunks, reading small fields only. Definitions
(the places where they go beyond the port's comments are listed in the batch-5 report):

- ``total``: every chunk, code files included.
- A chunk is *labellable* when it is a text chunk with positions; ``otherVersion`` counts only
  those, so the platform's "zero chunks at another version" barrier can be reached (code files
  and unpositioned chunks never get a version).
- ``otherVersion``: labellable chunks whose ``taxonomy_version`` is not the target (with an empty
  target, every labellable chunk).
- ``incrementalBacklog``: labellable chunks not at the target that an incremental job classifies:
  UNRESOLVED, PARTIALLY_ASSIGNED, ERROR **and NOT_LABELLED**.
- ``labelled``, ``errorChunks``, ``pieces``, ``unresolvedPieces``: over all chunks, any version.
  ``pieces`` are substantive pieces (all pieces minus administrative ones).
- ``documents`` and ``documentsBySource``: distinct source documents over all chunks, in total and
  per source (``airweave_system_metadata_source_name``; a chunk without one counts under
  ``unknown``). The operator console shows them as "documents" instead of entity counts.
- ``notLabelledChunks`` and ``notLabelledDocuments``: labellable chunks with no status or
  ``NOT_LABELLED`` whose own range is not empty (substantive; a whitespace-only own text of
  non-zero length still counts, because the text is not read here).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
from uuid import UUID

from airweave.domains.taxonomy.mapping import (
    ASSIGNED,
    ERROR,
    NOT_LABELLED,
    PARTIALLY_ASSIGNED,
)
from airweave.domains.taxonomy.relabel.chunk_store import ChunkStore
from airweave.domains.taxonomy.relabel.chunks import (
    CARRIED_OVER_STATUSES,
    F_SOURCE,
    INCREMENTAL_STATUSES,
    SCAN_FIELDS,
    ChunkRow,
)

SCAN_PAGE = 500


@dataclass
class ChunkStats:
    """The counts, with the port's camelCase names on the wire."""

    total: int = 0
    other_version: int = 0
    incremental_backlog: int = 0
    labelled: int = 0
    error_chunks: int = 0
    pieces: int = 0
    unresolved_pieces: int = 0
    documents_at_version: int = 0
    not_labelled_chunks: int = 0
    not_labelled_documents: int = 0
    documents: int = 0
    documents_by_source: Dict[str, int] = field(default_factory=dict)
    collection_id: Optional[str] = None
    # Not on the wire: the planner's view
    stampable: int = 0  # incremental: carried over, only the version is stamped
    labellable: int = 0
    versions: Counter = field(default_factory=Counter)  # labellable chunks per taxonomy version

    def to_port(self) -> Dict[str, Any]:
        """The JSON the platform's ``ChunkStats`` reads, plus a few extra fields."""
        return {
            "collectionId": self.collection_id,
            "total": self.total,
            "otherVersion": self.other_version,
            "incrementalBacklog": self.incremental_backlog,
            "labelled": self.labelled,
            "errorChunks": self.error_chunks,
            "pieces": self.pieces,
            "unresolvedPieces": self.unresolved_pieces,
            "documentsAtVersion": self.documents_at_version,
            "notLabelledChunks": self.not_labelled_chunks,
            "notLabelledDocuments": self.not_labelled_documents,
            "documents": self.documents,
            "documentsBySource": [
                {"source": source, "documents": n}
                for source, n in sorted(
                    self.documents_by_source.items(), key=lambda kv: (-kv[1], kv[0])
                )
            ],
            "labellableChunks": self.labellable,
            "stampableChunks": self.stampable,
        }

    def previous_version(self, target: str) -> Optional[str]:
        """The most common other version among labellable chunks."""
        others = [(n, v) for v, n in self.versions.items() if v and v != target]
        return max(others)[1] if others else None


async def collect_stats(
    store: ChunkStore, collection_id: UUID, target_version: str, page_size: int = SCAN_PAGE
) -> ChunkStats:
    """Scroll the whole collection once and count."""
    stats = ChunkStats(collection_id=str(collection_id))
    at_version: Set[str] = set()
    not_labelled_docs: Set[str] = set()
    by_source: Dict[str, Set[str]] = {}
    token: Optional[str] = None
    while True:
        rows, token = await store.scroll(collection_id, SCAN_FIELDS, page_size, token)
        for row in rows:
            _count(row, target_version, stats, at_version, not_labelled_docs)
            source = str(row.fields.get(F_SOURCE) or "unknown")
            by_source.setdefault(source, set()).add(row.document_key)
        if token is None:
            break
    stats.documents_at_version = len(at_version)
    stats.not_labelled_documents = len(not_labelled_docs)
    stats.documents_by_source = {source: len(ids) for source, ids in by_source.items()}
    stats.documents = len(set().union(*by_source.values())) if by_source else 0
    return stats


def _count(
    row: ChunkRow, target: str, stats: ChunkStats, at_version: Set[str], nl_docs: Set[str]
) -> None:
    stats.total += 1
    status, version = row.status, row.version
    if status in (ASSIGNED, PARTIALLY_ASSIGNED):
        stats.labelled += 1
    if status == ERROR:
        stats.error_chunks += 1
    admin = int(row.fields.get("taxonomy_admin_pieces") or 0)
    stats.pieces += max(0, int(row.fields.get("taxonomy_piece_count") or 0) - admin)
    stats.unresolved_pieces += int(row.fields.get("taxonomy_unresolved_pieces") or 0)
    if target and version == target:
        at_version.add(row.document_key)
    if not row.labellable:
        return
    stats.labellable += 1
    stats.versions[version or ""] += 1
    if version != target or not target:
        stats.other_version += 1
        if status in INCREMENTAL_STATUSES:
            stats.incremental_backlog += 1
        elif status in CARRIED_OVER_STATUSES and version:
            stats.stampable += 1
    if status == NOT_LABELLED and row.own_length > 0:
        stats.not_labelled_chunks += 1
        nl_docs.add(row.document_key)
