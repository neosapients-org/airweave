"""Accepting, reporting on and cancelling re-label jobs (the API side of A9).

The API never runs a job: it validates, checks the spend cap and writes a ``PENDING`` row. The
worker's ``RelabelRunner`` claims it. Replays are harmless: a start with a known idempotency key
returns that job, and an identical start while the collection's job is running returns it too.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from airweave.domains.taxonomy.mapping import PARTIALLY_ASSIGNED, UNRESOLVED
from airweave.domains.taxonomy.relabel.chunk_store import ChunkStore, decode_token
from airweave.domains.taxonomy.relabel.chunks import (
    F_CHUNK_INDEX,
    F_ENTITY_TYPE,
    F_SOURCE,
    F_TEXT,
    PAGE_FIELDS,
    SCAN_FIELDS,
    ChunkRow,
)
from airweave.domains.taxonomy.relabel.jobs import (
    MODE_FULL,
    MODE_INCREMENTAL,
    SCOPE_DIFFERING,
    SCOPE_NOT_LABELLED,
    ActiveJobExists,
    JobRepository,
    RelabelJob,
)
from airweave.domains.taxonomy.relabel.planning import (
    chunks_to_classify,
    effective_cap,
    estimate_usd,
)
from airweave.domains.taxonomy.relabel.stats import ChunkStats, collect_stats

_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


MAX_UNRESOLVED_PAGE = 500
UNRESOLVED_STATUSES = (UNRESOLVED, PARTIALLY_ASSIGNED)
UNRESOLVED_FIELDS = (
    *SCAN_FIELDS,
    F_TEXT,
    F_CHUNK_INDEX,
    F_SOURCE,
    F_ENTITY_TYPE,
    *PAGE_FIELDS,
)


def _encode_cursor(token: Optional[str]) -> Optional[str]:
    """Make the store's scroll token opaque and URL-safe."""
    if token is None:
        return None
    return base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")


def _decode_cursor(cursor: Optional[str]) -> Optional[str]:
    if not cursor:
        return None
    try:
        token = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        decode_token(token)
    except (ValueError, UnicodeDecodeError) as e:
        raise RelabelRefused(422, "bad_cursor", "the cursor is not usable") from e
    return token


def _unresolved_item(row: ChunkRow) -> Dict[str, Any]:
    f = row.fields
    start, end = row.own_range
    return {
        "chunk_id": row.doc_id,
        "schema": row.schema,
        "document_id": row.document_id,
        "chunk_index": f.get(F_CHUNK_INDEX),
        "source_name": f.get(F_SOURCE),
        "entity_type": f.get(F_ENTITY_TYPE),
        "text": row.own_text(),
        "taxonomy_status": row.status,
        "taxonomy_version": row.version,
        "unresolved_pieces": int(f.get("taxonomy_unresolved_pieces") or 0),
        "positions": {
            "char_start": int(f["chunk_char_start"]),
            "own_start": start,
            "char_end": end,
        },
        "page_from": f.get("page_from"),
        "page_to": f.get("page_to"),
    }


class RelabelRefused(Exception):
    """A start was refused; ``status`` is the HTTP status and ``code`` a stable machine name."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        """Keep the pieces the API turns into a response."""
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra

    def detail(self) -> Dict[str, Any]:
        """The response body's ``detail``."""
        return {"code": self.code, "message": self.message, **self.extra}


async def unresolved_page(
    store: ChunkStore,
    collection_id: UUID,
    limit: int,
    cursor: Optional[str] = None,
    taxonomy_version: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """One page of UNRESOLVED and PARTIALLY_ASSIGNED chunks (the endpoint and the propose job).

    The caller closes the store.
    """
    if not 1 <= limit <= MAX_UNRESOLVED_PAGE:
        raise RelabelRefused(422, "bad_limit", f"limit must be 1 to {MAX_UNRESOLVED_PAGE}")
    if taxonomy_version and not _VERSION.match(taxonomy_version):
        raise RelabelRefused(422, "bad_version", "taxonomy_version is malformed")
    token = _decode_cursor(cursor)
    items: List[Dict[str, Any]] = []
    while True:
        try:
            rows, token = await store.scroll(
                collection_id, UNRESOLVED_FIELDS, limit, token, UNRESOLVED_STATUSES
            )
        except ValueError as e:
            raise RelabelRefused(422, "bad_cursor", "the cursor is not usable") from e
        items += [
            _unresolved_item(r)
            for r in rows
            if r.labellable
            and r.status in UNRESOLVED_STATUSES
            and (not taxonomy_version or r.version == taxonomy_version)
        ]
        if token is None or len(items) >= limit:
            break
    return items, _encode_cursor(token)


@dataclass
class StartRequest:
    """What a start asks for."""

    organization_id: UUID
    collection_id: UUID
    collection_readable_id: str
    taxonomy_version: str
    mode: str = MODE_FULL
    scope: str = SCOPE_DIFFERING
    idempotency_key: Optional[str] = None
    spend_cap_usd: Optional[float] = None


class RelabelService:
    """Start, read and cancel jobs."""

    def __init__(
        self,
        repo: JobRepository,
        store_factory: Callable[[], ChunkStore],
        *,
        configured_cap_usd: float,
        usd_per_thousand: float,
        scan_page: int = 500,
    ) -> None:
        """``configured_cap_usd`` is ``RELABEL_SPEND_CAP_USD``, the approved ceiling per job."""
        self._repo = repo
        self._store_factory = store_factory
        self._cap = configured_cap_usd
        self._rate = usd_per_thousand
        self._scan_page = scan_page

    async def start(self, req: StartRequest) -> Tuple[RelabelJob, bool]:
        """Accept a job. Returns it and whether it was created now (False: a replay)."""
        self._validate(req)
        known = await self._replay(req)
        if known is not None:
            return known, False
        cap = effective_cap(req.spend_cap_usd, self._cap)
        store = self._store_factory()
        try:
            stats = await collect_stats(
                store, req.collection_id, req.taxonomy_version, self._scan_page
            )
        finally:
            await store.close()
        classify = chunks_to_classify(stats, req.mode, req.scope)
        estimate = estimate_usd(classify, self._rate)
        if cap <= 0 and classify > 0:  # a job that calls no embedder needs no approved spend
            raise RelabelRefused(
                422,
                "spend_cap_not_approved",
                "no spend cap is approved for re-label jobs (RELABEL_SPEND_CAP_USD is 0)",
                estimate_usd=estimate,
                cap_usd=cap,
            )
        if estimate > cap:
            raise RelabelRefused(
                422,
                "spend_cap_exceeded",
                f"estimated cost USD {estimate} for {classify} chunks is above the cap USD {cap}",
                estimate_usd=estimate,
                cap_usd=cap,
                chunks=classify,
            )
        stamped = stats.stampable if req.mode == MODE_INCREMENTAL else 0
        job = RelabelJob(
            id=uuid4(),
            organization_id=req.organization_id,
            collection_id=req.collection_id,
            collection_readable_id=req.collection_readable_id,
            target_version=req.taxonomy_version,
            mode=req.mode,
            scope=req.scope,
            idempotency_key=req.idempotency_key,
            previous_version=stats.previous_version(req.taxonomy_version),
            spend_cap_usd=cap,
            spend_estimate_usd=estimate,
            chunks_total=classify + stamped,
        )
        try:
            return await self._repo.create(job), True
        except ActiveJobExists as e:  # lost a race with another start
            return self._same_or_conflict(e.job, req), False

    async def statistics(self, collection_id: UUID, target_version: str) -> ChunkStats:
        """Chunk statistics against ``target_version`` (empty is allowed: bootstrap)."""
        if target_version and not _VERSION.match(target_version):
            raise RelabelRefused(422, "bad_version", "target_version is malformed")
        store = self._store_factory()
        try:
            return await collect_stats(store, collection_id, target_version, self._scan_page)
        finally:
            await store.close()

    async def unresolved(
        self,
        collection_id: UUID,
        limit: int,
        cursor: Optional[str] = None,
        taxonomy_version: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """One page of UNRESOLVED and PARTIALLY_ASSIGNED chunks, read-only.

        ``limit`` is a target: a page returns every row of the last Vespa page it needed, so no
        row is ever skipped between pages. Text chunks only; the text is the chunk's own text.
        """
        store = self._store_factory()
        try:
            return await unresolved_page(store, collection_id, limit, cursor, taxonomy_version)
        finally:
            await store.close()

    async def get(self, job_id: UUID, organization_id: UUID) -> Optional[RelabelJob]:
        """The job, only inside the caller's organization."""
        return await self._repo.get(job_id, organization_id)

    async def cancel(self, job_id: UUID, organization_id: UUID) -> Optional[RelabelJob]:
        """Ask a job to stop; a job nobody is running is cancelled at once."""
        if await self._repo.get(job_id, organization_id) is None:
            return None
        return await self._repo.request_cancel(job_id)

    # ---------------------------------------------------------------- checks

    @staticmethod
    def _validate(req: StartRequest) -> None:
        if not req.taxonomy_version or not _VERSION.match(req.taxonomy_version):
            raise RelabelRefused(422, "bad_version", "taxonomy_version is missing or malformed")
        if req.mode not in (MODE_FULL, MODE_INCREMENTAL):
            raise RelabelRefused(422, "bad_mode", "mode must be 'full' or 'incremental'")
        if req.scope not in (SCOPE_DIFFERING, SCOPE_NOT_LABELLED):
            raise RelabelRefused(422, "bad_scope", "scope must be 'differing' or 'not_labelled'")
        if req.scope == SCOPE_NOT_LABELLED and req.mode != MODE_FULL:
            raise RelabelRefused(
                422, "bad_mode", "scope 'not_labelled' labels chunks and has no incremental mode"
            )
        if req.idempotency_key is not None and not _KEY.match(req.idempotency_key):
            raise RelabelRefused(422, "bad_key", "idempotency_key is malformed")
        if req.spend_cap_usd is not None and req.spend_cap_usd < 0:
            raise RelabelRefused(422, "bad_cap", "spend_cap_usd cannot be negative")

    async def _replay(self, req: StartRequest) -> Optional[RelabelJob]:
        if req.idempotency_key:
            known = await self._repo.find_by_key(req.collection_id, req.idempotency_key)
            if known is not None:
                if not self._same(known, req):
                    raise RelabelRefused(
                        409,
                        "idempotency_key_reused",
                        "this idempotency key started a job with different parameters",
                        relabel_job_id=str(known.id),
                    )
                return known
        active = await self._repo.find_active(req.collection_id)
        if active is not None:
            return self._same_or_conflict(active, req)
        return None

    def _same_or_conflict(self, job: RelabelJob, req: StartRequest) -> RelabelJob:
        if self._same(job, req):
            return job
        raise RelabelRefused(
            409,
            "job_active",
            f"collection {req.collection_readable_id} already has a running job "
            f"for {job.target_version!r}",
            relabel_job_id=str(job.id),
        )

    @staticmethod
    def _same(job: RelabelJob, req: StartRequest) -> bool:
        return (job.target_version, job.mode, job.scope) == (
            req.taxonomy_version,
            req.mode,
            req.scope,
        )
