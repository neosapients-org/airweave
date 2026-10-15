"""Chunk records as the re-label job, the statistics and the unresolved scroll read them.

The fields are the Vespa field names (contract 4.1 and 4.2). A chunk is *labellable* when it can
carry labels: a text chunk (not a code file) with positions (``CHUNKER_MODE=sentence_pack``).
Everything else is counted but never classified or written (contract 4.2, "Labels need
positions").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from airweave.domains.taxonomy.mapping import (
    ADMINISTRATIVE,
    ASSIGNED,
    ERROR,
    NOT_LABELLED,
    PARTIALLY_ASSIGNED,
    UNRESOLVED,
    ChunkLabels,
)

CODE_SCHEMA = "code_file_entity"

# Fields every scan reads. The text is read only when a chunk is about to be classified.
F_HASH = "airweave_system_metadata_hash"
F_DOCUMENT = "airweave_system_metadata_original_entity_id"
F_CHUNK_INDEX = "airweave_system_metadata_chunk_index"
F_TEXT = "textual_representation"
POSITION_FIELDS = ("chunk_char_start", "chunk_char_end", "chunk_own_start", "chunk_body_offset")
PAGE_FIELDS = ("page_from", "page_to")
LABEL_FIELDS = (
    "taxonomy_status",
    "taxonomy_version",
    "taxonomy_piece_count",
    "taxonomy_admin_pieces",
    "taxonomy_unresolved_pieces",
)
SCAN_FIELDS = (F_HASH, F_DOCUMENT, *POSITION_FIELDS, *LABEL_FIELDS)

# Statuses an incremental job classifies again (the labels they hold may now be improvable)
INCREMENTAL_STATUSES = frozenset({UNRESOLVED, PARTIALLY_ASSIGNED, ERROR, NOT_LABELLED})
# Statuses whose labels stay valid when the target only adds categories
CARRIED_OVER_STATUSES = frozenset({ASSIGNED, ADMINISTRATIVE})


@dataclass
class ChunkRow:
    """One chunk read from Vespa: where it lives and the fields that were requested."""

    schema: str
    doc_id: str  # the user part of the Vespa document id
    fields: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> Optional[str]:
        """The document's content hash, the same on all its chunks (contract 3.2)."""
        return self.fields.get(F_HASH)

    @property
    def document_id(self) -> str:
        """The source document's id, from the system metadata (never parsed from the Vespa id)."""
        return self.fields.get(F_DOCUMENT) or self.doc_id

    @property
    def status(self) -> str:
        """The label status; a chunk without one reads as ``NOT_LABELLED`` (contract 4.2)."""
        return self.fields.get("taxonomy_status") or NOT_LABELLED

    @property
    def version(self) -> Optional[str]:
        """The taxonomy version that produced the labels, if any."""
        return self.fields.get("taxonomy_version") or None

    @property
    def has_positions(self) -> bool:
        """True for text chunks written with ``CHUNKER_MODE=sentence_pack``."""
        f = self.fields
        return all(
            f.get(k) is not None for k in ("chunk_char_start", "chunk_char_end", "chunk_own_start")
        )

    @property
    def labellable(self) -> bool:
        """A text chunk with positions; code files and unpositioned chunks never get labels."""
        return self.schema != CODE_SCHEMA and self.has_positions

    @property
    def own_range(self) -> tuple[int, int]:
        """``[chunk_own_start, chunk_char_end)`` in the document text."""
        return int(self.fields["chunk_own_start"]), int(self.fields["chunk_char_end"])

    @property
    def own_length(self) -> int:
        """Length of the chunk's own text; 0 for a chunk that only overlaps its neighbours."""
        start, end = self.own_range
        return max(0, end - start)

    def own_text(self) -> str:
        """The chunk's own text: no header, no overlap (contract 3.2, C16)."""
        start, _ = self.own_range
        body = (self.fields.get(F_TEXT) or "")[int(self.fields.get("chunk_body_offset") or 0) :]
        return body[start - int(self.fields["chunk_char_start"]) :]


def label_fields(labels: ChunkLabels, version: str) -> Dict[str, Any]:
    """The label fields of a classified chunk, exactly what the sync writes (contract 4.2)."""
    return {
        "taxonomy_status": labels.taxonomy_status,
        "topic_ids": dict(labels.topic_ids),
        "broad_family_ids": list(labels.broad_family_ids),
        "taxonomy_spans": labels.taxonomy_spans,
        "taxonomy_version": version,
        "taxonomy_piece_count": labels.taxonomy_piece_count,
        "taxonomy_admin_pieces": labels.taxonomy_admin_pieces,
        "taxonomy_unresolved_pieces": labels.taxonomy_unresolved_pieces,
    }
