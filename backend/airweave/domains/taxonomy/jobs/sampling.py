"""Sampling a collection's stored chunk text for a discover or propose job.

The text is read from Vespa into memory for the length of the job; nothing is copied anywhere
else. The sample is deterministic (documents ordered by a hash of the candidate version and the
document id), so a job that runs again picks the same documents.
"""

from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable, Dict, List, Optional
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
    """The collection has less text than a job needs (too few documents, or too few chunks)."""


def _order_key(collection_id: UUID, document_key: str) -> str:
    """Seeded by the collection, not the job: a re-run samples the same documents."""
    return hashlib.sha256(f"{collection_id}:{document_key}".encode()).hexdigest()


def to_package_chunk(row: ChunkRow) -> Dict[str, Any]:
    """A stored chunk as the classifier package reads it: own text, source as the group.

    ``chunk_id`` keeps the schema (document ids repeat across Vespa schemas) and ``document_id``
    is the document's identity across the collection (sync, schema, source id).
    """
    return {
        "chunk_id": f"{row.schema}/{row.doc_id}",
        "document_id": row.document_key,
        "text": row.own_text(),
        "metadata": {
            "source": row.fields.get(F_SOURCE) or "n/a",
            "entity_type": row.fields.get(F_ENTITY_TYPE) or "n/a",
        },
    }


OnPage = Optional[Callable[[], Awaitable[None]]]


async def sample_documents(
    store: ChunkStore,
    collection_id: UUID,
    documents: int,
    max_chunks: int,
    min_documents: int,
    min_chunks: int = 0,
    on_page: OnPage = None,
) -> List[Dict[str, Any]]:
    """Chunks of a deterministic sample of the collection's documents (text chunks only).

    Two scrolls. The first counts each document's substantive text chunks (small fields only),
    which fixes the sample before any text is read: documents in a hash order until ``max_chunks``
    chunks, a document that alone exceeds ``max_chunks`` left out. The second reads the text of
    exactly those documents, so memory holds at most ``max_chunks`` chunks of text. ``on_page`` is
    awaited after every page, so a cancel or a lost lease stops the scan.
    """
    counts = await _substantive_documents(store, collection_id, on_page)
    if len(counts) < min_documents:
        raise TooFewDocuments(
            f"{len(counts)} substantive documents, a job needs at least {min_documents}"
        )
    plan = _plan(collection_id, counts, documents, max_chunks)
    if len(plan) < min_documents:
        raise TooFewDocuments(f"the sample holds fewer than {min_documents} documents")
    by_document = await _read_documents(store, collection_id, plan, max_chunks, on_page)
    sample: List[Dict[str, Any]] = []
    for document in sorted(by_document, key=plan.__getitem__):
        sample.extend(sorted(by_document[document], key=lambda c: c["chunk_id"]))
    if len({c["document_id"] for c in sample}) < min_documents:
        raise TooFewDocuments(f"the sample holds fewer than {min_documents} documents")
    if len(sample) < min_chunks:
        # Checked before anything is paid for: the classifier's clustering needs hundreds of pieces
        # of text, and on a handful it fails late with an unhelpful error.
        raise TooFewDocuments(
            f"not enough text yet: {len(sample)} chunks, a discovery needs at least {min_chunks}. "
            "Add more documents and try again"
        )
    return sample


def _plan(
    collection_id: UUID, counts: Dict[str, int], documents: int, max_chunks: int
) -> Dict[str, int]:
    """Document key -> rank, for the documents whose chunks fit in ``max_chunks`` together."""
    plan: Dict[str, int] = {}
    total = 0
    for key in sorted(counts, key=lambda d: _order_key(collection_id, d)):
        if len(plan) >= documents:
            break
        if counts[key] > max_chunks or total + counts[key] > max_chunks:
            continue  # too big on its own, or does not fit next to the ones chosen
        plan[key] = len(plan)
        total += counts[key]
    return plan


async def _substantive_documents(
    store: ChunkStore, collection_id: UUID, on_page: OnPage
) -> Dict[str, int]:
    """Documents with at least one text chunk that has own text, and how many such chunks."""
    counts: Dict[str, int] = {}
    token: Optional[str] = None
    while True:
        rows, token = await store.scroll(collection_id, SCAN_FIELDS, PAGE, token)
        for r in rows:
            if r.labellable and r.own_length > 0:
                counts[r.document_key] = counts.get(r.document_key, 0) + 1
        if on_page is not None:
            await on_page()
        if token is None:
            return counts


async def _read_documents(
    store: ChunkStore,
    collection_id: UUID,
    wanted: Dict[str, int],
    max_chunks: int,
    on_page: OnPage,
) -> Dict[str, List[Dict[str, Any]]]:
    """The package-shaped chunks of the wanted documents, at most ``max_chunks`` of them."""
    by_document: Dict[str, List[Dict[str, Any]]] = {}
    held = 0
    token: Optional[str] = None
    fields = (*SCAN_FIELDS, F_TEXT, F_SOURCE, F_ENTITY_TYPE)
    while True:
        rows, token = await store.scroll(collection_id, fields, PAGE, token)
        for row in rows:
            if held >= max_chunks:
                break  # the collection grew since the plan was made: the bound still holds
            if row.labellable and row.document_key in wanted and row.own_text().strip():
                by_document.setdefault(row.document_key, []).append(to_package_chunk(row))
                held += 1
        if on_page is not None:
            await on_page()
        if token is None or held >= max_chunks:
            return by_document
