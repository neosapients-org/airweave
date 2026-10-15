"""Sampling a collection's stored chunk text for a discover or propose job.

The text is read from Vespa into memory for the length of the job; nothing is copied anywhere
else. The sample is deterministic (documents ordered by a hash of the candidate version and the
document id), so a job that runs again picks the same documents.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from airweave.domains.taxonomy.relabel.chunk_store import ChunkStore
from airweave.domains.taxonomy.relabel.chunks import (
    F_ENTITY_TYPE,
    F_SOURCE,
    F_TEXT,
    SCAN_FIELDS,
    ChunkRow,
)

PAGE = 500


class TooFewDocuments(Exception):
    """The collection has fewer substantive documents than a job needs."""


def _order_key(candidate_version: str, document_id: str) -> str:
    return hashlib.sha256(f"{candidate_version}:{document_id}".encode()).hexdigest()


def to_package_chunk(row: ChunkRow) -> Dict[str, Any]:
    """A stored chunk as the classifier package reads it: own text, source as the group."""
    return {
        "chunk_id": row.doc_id,
        "document_id": row.document_id,
        "text": row.own_text(),
        "metadata": {
            "source": row.fields.get(F_SOURCE) or "n/a",
            "entity_type": row.fields.get(F_ENTITY_TYPE) or "n/a",
        },
    }


async def sample_documents(
    store: ChunkStore,
    collection_id: UUID,
    candidate_version: str,
    documents: int,
    max_chunks: int,
    min_documents: int,
) -> List[Dict[str, Any]]:
    """Chunks of a deterministic sample of the collection's documents (text chunks only).

    Two scrolls: the first lists the documents that have a substantive text chunk (small
    fields), the second reads the text of the chosen documents. Whole documents are kept until
    ``max_chunks`` is reached.
    """
    seen = await _substantive_documents(store, collection_id)
    if len(seen) < min_documents:
        raise TooFewDocuments(
            f"{len(seen)} substantive documents, a job needs at least {min_documents}"
        )
    chosen = sorted(seen, key=lambda d: _order_key(candidate_version, d))[:documents]
    rank = {d: i for i, d in enumerate(chosen)}
    by_document = await _read_documents(store, collection_id, rank)
    sample: List[Dict[str, Any]] = []
    for document in sorted(by_document, key=rank.__getitem__):
        chunks = sorted(by_document[document], key=lambda c: c["chunk_id"])
        if sample and len(sample) + len(chunks) > max_chunks:
            break
        sample.extend(chunks)
    if len({c["document_id"] for c in sample}) < min_documents:
        raise TooFewDocuments(f"the sample holds fewer than {min_documents} documents")
    return sample


async def _substantive_documents(store: ChunkStore, collection_id: UUID) -> Set[str]:
    """Documents with at least one text chunk that has own text (small fields only)."""
    seen: Set[str] = set()
    token: Optional[str] = None
    while True:
        rows, token = await store.scroll(collection_id, SCAN_FIELDS, PAGE, token)
        seen.update(r.document_id for r in rows if r.labellable and r.own_length > 0)
        if token is None:
            return seen


async def _read_documents(
    store: ChunkStore, collection_id: UUID, wanted: Dict[str, int]
) -> Dict[str, List[Dict[str, Any]]]:
    """The package-shaped chunks of the wanted documents."""
    by_document: Dict[str, List[Dict[str, Any]]] = {}
    token: Optional[str] = None
    fields = (*SCAN_FIELDS, F_TEXT, F_SOURCE, F_ENTITY_TYPE)
    while True:
        rows, token = await store.scroll(collection_id, fields, PAGE, token)
        for row in rows:
            if row.labellable and row.document_id in wanted and row.own_text().strip():
                by_document.setdefault(row.document_id, []).append(to_package_chunk(row))
        if token is None:
            return by_document
