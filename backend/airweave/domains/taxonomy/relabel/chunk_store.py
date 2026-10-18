"""Reading and conditionally updating a collection's chunks in Vespa.

``ChunkStore`` is the seam the re-label job, the statistics and the unresolved scroll work
through. ``VespaChunkStore`` talks to Vespa's document API; the tests use an in-memory store that
honours the same condition.

Verified on a real Vespa 8 (``tests/integration/vespa/test_relabel_contract.py``): the visit
parameters, the document selection and the test-and-set condition work as written below. Measured
details: a failed condition answers **412**, and so does a **missing document** ("Document does
not exist"), so a chunk deleted meanwhile is reported as ``CONDITION_FAILED`` (skipped, counted);
404 is kept as an outcome in case another Vespa version answers it. An update WITHOUT a condition
on a missing document answers 200, which is why every update here carries one.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import quote
from uuid import UUID

import httpx

from airweave.core.config import settings
from airweave.domains.taxonomy.relabel.chunks import F_HASH, ChunkRow
from airweave.platform.destinations.vespa.config import ALL_VESPA_SCHEMAS

APPLIED = "applied"
CONDITION_FAILED = "condition_failed"  # a sync replaced the chunk: skipped, not an error
NOT_FOUND = "not_found"  # a 404; Vespa 8 answers 412 for a deleted chunk

_HASH = re.compile(r"^[A-Za-z0-9_.:+/=-]{1,200}$")


class ChunkStore(Protocol):
    """A collection's chunks: a resumable scroll and a conditional partial update."""

    async def scroll(
        self,
        collection_id: UUID,
        fields: Sequence[str],
        page_size: int,
        continuation: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> Tuple[List[ChunkRow], Optional[str]]:
        """One page of chunks (all schemas) and the token for the next, ``None`` at the end.

        ``statuses`` keeps only chunks whose ``taxonomy_status`` is one of them. A token from an
        earlier scroll resumes it; an unusable token is the caller's cue to start again.
        """
        ...

    async def update(self, chunk: ChunkRow, fields: Mapping[str, Any], expected_hash: str) -> str:
        """Assign ``fields`` on the chunk only if its content hash is still ``expected_hash``.

        Returns ``APPLIED``, ``CONDITION_FAILED`` or ``NOT_FOUND``; raises on any other failure.
        """
        ...

    async def close(self) -> None:
        """Release connections."""
        ...


def encode_token(schema_index: int, vespa_token: Optional[str]) -> str:
    """Our scroll token: which schema we are in and Vespa's continuation inside it."""
    return json.dumps([schema_index, vespa_token])


def decode_token(token: Optional[str]) -> Tuple[int, Optional[str]]:
    """Inverse of ``encode_token``; raises ``ValueError`` for anything else."""
    if token is None:
        return 0, None
    try:
        index, inner = json.loads(token)
        if not isinstance(index, int) or index < 0 or not (inner is None or isinstance(inner, str)):
            raise ValueError
    except (ValueError, TypeError) as e:
        raise ValueError("unusable scroll token") from e
    return index, inner


class VespaChunkStore:
    """Vespa document API: visit for scrolling, PUT with ``condition`` for the test-and-set."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0) -> None:
        """Use ``VESPA_URL:VESPA_PORT`` unless a base URL is given."""
        self._base = base_url or f"{settings.VESPA_URL}:{settings.VESPA_PORT}"
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def scroll(
        self,
        collection_id: UUID,
        fields: Sequence[str],
        page_size: int,
        continuation: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> Tuple[List[ChunkRow], Optional[str]]:
        """Visit the collection's chunks, one schema after the other, page by page."""
        index, inner = decode_token(continuation)
        schemas = list(ALL_VESPA_SCHEMAS)
        if continuation is not None and index >= len(schemas):
            # A token this store never issued (it returns None after the last schema): reading it
            # as "finished" would silently truncate a walk, so it is refused like any bad cursor.
            raise ValueError("unusable scroll token: no such schema")
        while index < len(schemas):
            schema = schemas[index]
            url = (
                f"{self._base}/document/v1/airweave/{schema}/docid"
                f"?selection={quote(_selection(schema, collection_id, statuses), safe='')}"
                f"&wantedDocumentCount={page_size}"
                f"&fieldSet={quote(schema + ':' + ','.join(fields), safe='')}"
            )
            if inner:
                url += f"&continuation={quote(inner, safe='')}"
            resp = await self._client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Vespa visit failed for {schema}: {resp.status_code} {resp.text[:200]}"
                )
            data = resp.json()
            rows = [_row(schema, d) for d in data.get("documents", [])]
            inner = data.get("continuation")
            if inner is None:
                index, inner = index + 1, None
            next_token = encode_token(index, inner) if index < len(schemas) else None
            if rows or next_token is None:
                return rows, next_token
        return [], None

    async def update(self, chunk: ChunkRow, fields: Mapping[str, Any], expected_hash: str) -> str:
        """PUT an ``assign`` update with a document-selection condition on the content hash."""
        if not _HASH.match(expected_hash):
            raise ValueError("refusing to build a condition from an unexpected hash value")
        condition = f'{chunk.schema}.{F_HASH}=="{expected_hash}"'
        url = (
            f"{self._base}/document/v1/airweave/{chunk.schema}/docid/"
            f"{quote(chunk.doc_id, safe='')}?condition={quote(condition, safe='')}&create=false"
        )
        body = {"fields": {name: {"assign": value} for name, value in fields.items()}}
        resp = await self._client.put(url, json=body)
        if resp.status_code == 200:
            return APPLIED
        if resp.status_code == 412:
            return CONDITION_FAILED
        if resp.status_code == 404:
            return NOT_FOUND
        raise RuntimeError(
            f"Vespa conditional update failed for {chunk.doc_id}: "
            f"{resp.status_code} {resp.text[:200]}"
        )


def _selection(schema: str, collection_id: UUID, statuses: Optional[Sequence[str]]) -> str:
    clause = f"{schema}.airweave_system_metadata_collection_id=='{collection_id}'"
    if statuses:
        wanted = " or ".join(f'{schema}.taxonomy_status=="{s}"' for s in statuses)
        clause = f"{clause} and ({wanted})"
    return clause


def _row(schema: str, document: Dict[str, Any]) -> ChunkRow:
    full_id = document.get("id", "")
    doc_id = full_id.split("::", 1)[1] if "::" in full_id else full_id
    return ChunkRow(schema=schema, doc_id=doc_id, fields=dict(document.get("fields") or {}))
