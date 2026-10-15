"""Fakes for the re-label tests.

A Vespa that honours the test-and-set condition, a resolver, a classifier pool and an event bus.
Invented data only (Acme, Globex).
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.mapping import RegistryMap
from airweave.domains.taxonomy.relabel.chunk_store import (
    APPLIED,
    CONDITION_FAILED,
    NOT_FOUND,
    decode_token,
    encode_token,
)
from airweave.domains.taxonomy.relabel.chunks import F_DOCUMENT, F_HASH, ChunkRow

COLLECTION = UUID(int=11)
OTHER_COLLECTION = UUID(int=12)
ORG = UUID(int=1)
READABLE = "neo-acme"
OLD, NEW = "acme-2026.10", "acme-2026.11"
HEADER = "Acme Q3.pdf · google_drive"

REGISTRY = RegistryMap(
    NEW,
    {"CAT001": "acme:credit.loan_quality", "CAT002": "acme:revenue.growth"},
    {"BF01": "acme:credit"},
)


def taxonomy(version: str = NEW) -> ResolvedTaxonomy:
    return ResolvedTaxonomy(
        READABLE,
        "t-acme",
        version,
        "a" * 64,
        4,
        "calibrated",
        None,
        Path("/tmp/none"),
        {},
        REGISTRY,
    )


def chunk(
    n: int,
    *,
    doc: str = "doc-1",
    schema: str = "file_entity",
    status: Optional[str] = None,
    version: Optional[str] = None,
    hash_: Optional[str] = "h1",
    body: str = "Late loan payments rose sharply this quarter.",
    positions: bool = True,
    collection: UUID = COLLECTION,
    pieces: int = 1,
    unresolved: int = 0,
    topic_ids: Optional[Dict[str, int]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """(schema, doc id, fields) of one stored chunk."""
    fields: Dict[str, Any] = {
        "airweave_system_metadata_collection_id": str(collection),
        F_DOCUMENT: doc,
        F_HASH: hash_,
        "textual_representation": f"{HEADER}\n{body}",
        "chunk_body_offset": len(HEADER) + 1,
        "topic_ids": topic_ids or {},
    }
    if positions:
        fields.update(chunk_char_start=0, chunk_own_start=0, chunk_char_end=len(body))
    if status:
        fields["taxonomy_status"] = status
    if version:
        fields["taxonomy_version"] = version
    if status and status not in ("ERROR", "NOT_LABELLED"):  # those carry no counts
        fields["taxonomy_piece_count"] = pieces
        fields["taxonomy_unresolved_pieces"] = unresolved
        fields["taxonomy_admin_pieces"] = 0
    return schema, f"{schema}_{doc}__chunk_{n}", fields


class FakeChunkStore:
    """In-memory Vespa: one dict of documents, a resumable scroll, and the same condition rule."""

    def __init__(self, chunks: Sequence[Tuple[str, str, Dict[str, Any]]] = ()) -> None:
        self.docs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for schema, doc_id, fields in chunks:
            self.docs[(schema, doc_id)] = copy.deepcopy(fields)
        self.updates: List[Tuple[str, Mapping[str, Any], str]] = []
        self.before_update: Optional[Callable[[str, str], Awaitable[None]]] = None
        self.fail_updates: Optional[Exception] = None
        self.closed = False
        self.scrolls = 0

    def keys(self) -> List[Tuple[str, str]]:
        return sorted(self.docs)

    async def scroll(self, collection_id, fields, page_size, continuation=None, statuses=None):
        self.scrolls += 1
        index, _ = decode_token(continuation)
        keys = [
            k
            for k in self.keys()
            if self.docs[k].get("airweave_system_metadata_collection_id") == str(collection_id)
            and (not statuses or self.docs[k].get("taxonomy_status") in statuses)
        ]
        page = keys[index : index + page_size]
        rows = [
            ChunkRow(
                k[0], k[1], {f: copy.deepcopy(self.docs[k][f]) for f in fields if f in self.docs[k]}
            )
            for k in page
        ]
        nxt = index + page_size
        return rows, (encode_token(nxt, None) if nxt < len(keys) else None)

    async def update(self, chunk: ChunkRow, fields, expected_hash: str) -> str:
        if self.before_update:
            await self.before_update(chunk.schema, chunk.doc_id)
        if self.fail_updates:
            raise self.fail_updates
        doc = self.docs.get((chunk.schema, chunk.doc_id))
        if doc is None:
            return NOT_FOUND
        if doc.get(F_HASH) != expected_hash:
            return CONDITION_FAILED
        self.updates.append((chunk.doc_id, dict(fields), expected_hash))
        doc.update(copy.deepcopy(dict(fields)))
        return APPLIED

    async def close(self) -> None:
        self.closed = True

    def get(self, schema: str, doc_id: str) -> Dict[str, Any]:
        return self.docs[(schema, doc_id)]


class FakeResolver:
    def __init__(self, result: Any = None) -> None:
        self.result = taxonomy() if result is None else result
        self.calls: List[Tuple[str, bool]] = []

    async def resolve(self, readable_id: str, fresh: bool = False):
        self.calls.append((readable_id, fresh))
        if isinstance(self.result, Exception):
            raise self.result
        return None if self.result == "none" else self.result


class FakePool:
    """Labels every row CAT001 over its whole text; can raise, and records what it was sent."""

    def __init__(self, fail: Optional[BaseException] = None, drop: Sequence[str] = ()) -> None:
        self.fail, self.drop = fail, set(drop)
        self.rows: List[Dict[str, Any]] = []
        self.calls = 0
        self.on_call: Optional[Callable[[], None]] = None

    async def classify(self, tax, rows, timeout):
        self.calls += 1
        if self.on_call:
            self.on_call()
        self.rows.extend(rows)
        if self.fail is not None:
            raise self.fail
        return [
            {
                "chunk_id": r["chunk_id"],
                "status": "ASSIGNED",
                "topics": [
                    {"category_id": "CAT001", "broad_family_id": "BF01", "supporting_children": 1}
                ],
                "taxonomy_children": [
                    {
                        "taxonomy_chunk_id": "t",
                        "status": "ASSIGNED",
                        "char_start": 0,
                        "char_end": len(r["text"]),
                        "topic_ids": ["CAT001"],
                    }
                ],
            }
            for r in rows
            if r["chunk_id"] not in self.drop
        ]


class FakeBus:
    def __init__(self) -> None:
        self.events: List[Any] = []
        self.fail = False

    async def publish(self, event) -> None:
        if self.fail:
            raise RuntimeError("bus down")
        self.events.append(event)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 15, 9, 0, 0)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def new_job(**kw):
    from airweave.domains.taxonomy.relabel.jobs import RelabelJob

    defaults = {
        "id": uuid4(),
        "organization_id": ORG,
        "collection_id": COLLECTION,
        "collection_readable_id": READABLE,
        "target_version": NEW,
        "spend_cap_usd": 10.0,
        "previous_version": OLD,
    }
    defaults.update(kw)
    return RelabelJob(**defaults)
