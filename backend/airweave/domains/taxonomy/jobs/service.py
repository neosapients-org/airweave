"""Accepting, reporting on and cancelling discover and propose jobs (the API side).

The API never runs a job: it validates, applies the spend cap and writes a ``PENDING`` row, then
starts a Temporal workflow on the dedicated queue through ``enqueue``. Replays are harmless: a
known idempotency key returns its job; the same candidate version is never started twice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
from uuid import UUID, uuid4

from airweave.core.logging import logger
from airweave.domains.taxonomy.artifacts import is_safe_name
from airweave.domains.taxonomy.jobs.records import (
    KIND_DISCOVER,
    KIND_PROPOSE,
    ActiveTaxonomyJob,
    TaxonomyJobRecord,
    TaxonomyJobRepository,
)
from airweave.domains.taxonomy.relabel.service import RelabelRefused

_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MAX_SAMPLE = 5000

Enqueue = Callable[[TaxonomyJobRecord], Awaitable[None]]


@dataclass
class JobRequest:
    """What a start asks for."""

    kind: str
    organization_id: UUID
    collection_id: UUID
    collection_readable_id: str
    tenant_id: str
    candidate_version: str
    idempotency_key: Optional[str] = None
    spend_cap_usd: Optional[float] = None
    sample_size: Optional[int] = None
    corpus_description: Optional[str] = None


def resolve_tenant(active_tenant: Optional[str], tenant_id: Optional[str]) -> str:
    """The tenant whose prefix a discovery writes its candidate to.

    A first discovery takes the tenant the caller names (Meridian sends it); later ones take the
    collection's active taxonomy's, and a name given then must match it. Airweave does not check
    that the caller owns the tenant: Neo calls Airweave with one service key, and the endpoints
    already require the admin permission. The candidate only ever goes under ``candidates/``,
    is never written over, and Meridian registers only the candidates of jobs it started, so a
    wrong tenant name can at most leave files nobody reads. (Writing candidates under the
    collection instead of the tenant would remove the question entirely.)

    Raises:
        RelabelRefused: 409 when a named tenant differs from the active taxonomy's, 422 when a
            first discovery names no tenant.
    """
    if tenant_id:
        if active_tenant and active_tenant != tenant_id:
            raise RelabelRefused(
                409,
                "tenant_conflict",
                "the collection's active taxonomy names a different tenant",
            )
        return tenant_id
    if active_tenant:
        return active_tenant
    raise RelabelRefused(
        422,
        "tenant_required",
        "a first discovery needs tenant_id: the collection has no active taxonomy to take it from",
    )


class TaxonomyJobService:
    """Start, read and cancel jobs."""

    def __init__(
        self,
        repo: TaxonomyJobRepository,
        enqueue: Enqueue,
        *,
        cap_usd: float,
    ) -> None:
        """``cap_usd`` is ``TAXONOMY_SPEND_CAP_USD``, the ceiling of one job of either kind.

        Raises ``ValueError`` for a ceiling that is not a finite amount from 0 up.
        """
        if not math.isfinite(cap_usd) or cap_usd < 0:
            raise ValueError(f"invalid taxonomy job spend cap {cap_usd!r}")
        self._repo = repo
        self._enqueue = enqueue
        self._cap = cap_usd

    async def start(self, req: JobRequest) -> Tuple[TaxonomyJobRecord, bool]:
        """Accept a job; returns it and whether it was created now (False: a replay)."""
        self._validate(req)
        known = await self._replay(req)
        if known is not None:
            if (
                known.state == "PENDING"
            ):  # accepted, but its dispatch may never have reached Temporal
                await self._dispatch(known)
            return known, False
        ceiling = self._cap
        cap = ceiling if req.spend_cap_usd is None else min(req.spend_cap_usd, ceiling)
        if cap <= 0:
            raise RelabelRefused(
                422,
                "spend_cap_not_approved",
                f"no spend cap is approved for {req.kind} jobs (TAXONOMY_SPEND_CAP_USD is 0)",
                cap_usd=cap,
            )
        job = TaxonomyJobRecord(
            id=uuid4(),
            organization_id=req.organization_id,
            kind=req.kind,
            collection_id=req.collection_id,
            collection_readable_id=req.collection_readable_id,
            tenant_id=req.tenant_id,
            candidate_version=req.candidate_version,
            idempotency_key=req.idempotency_key,
            params={
                k: v
                for k, v in (
                    ("sample_size", req.sample_size),
                    ("corpus_description", req.corpus_description),
                    ("fingerprint", fingerprint(req)),
                )
                if v is not None
            },
            spend_cap_usd=cap,
        )
        try:
            created = await self._repo.create(job)
        except ActiveTaxonomyJob as e:
            return self._same_or_conflict(e.job, req), False
        await self._dispatch(created)
        return created, True

    async def _dispatch(self, job: TaxonomyJobRecord) -> None:
        """Start the job's workflow.

        A failure is not the request's: the accepted row stays PENDING and the jobs worker's
        reconciler (or a replay of this request) starts it.
        """
        try:
            await self._enqueue(job)
        except Exception as e:  # noqa: BLE001 - Temporal is down or unreachable
            logger.warning(
                f"[TaxonomyJob] job {job.id} accepted, but its workflow did not start "
                f"({type(e).__name__}: {e}); the reconciler will start it"
            )

    async def get(self, job_id: UUID, organization_id: UUID) -> Optional[TaxonomyJobRecord]:
        """The job, only inside the caller's organization."""
        return await self._repo.get(job_id, organization_id)

    async def cancel(self, job_id: UUID, organization_id: UUID) -> Optional[TaxonomyJobRecord]:
        """Ask a job to stop; one nobody runs is cancelled at once."""
        if await self._repo.get(job_id, organization_id) is None:
            return None
        return await self._repo.request_cancel(job_id)

    # ---------------------------------------------------------------- checks

    @staticmethod
    def _validate(req: JobRequest) -> None:
        if req.kind not in (KIND_DISCOVER, KIND_PROPOSE):
            raise RelabelRefused(422, "bad_kind", "kind must be discover or propose")
        if not req.candidate_version or not _VERSION.match(req.candidate_version):
            raise RelabelRefused(422, "bad_version", "candidate_version is missing or malformed")
        if not req.idempotency_key or not _KEY.match(req.idempotency_key):
            raise RelabelRefused(
                422, "bad_key", "idempotency_key is required and must be well formed"
            )
        if req.spend_cap_usd is not None and (
            not math.isfinite(req.spend_cap_usd) or req.spend_cap_usd < 0
        ):
            raise RelabelRefused(422, "bad_cap", "spend_cap_usd must be a finite amount, 0 or more")
        if not is_safe_name(req.tenant_id):
            raise RelabelRefused(422, "bad_tenant", "tenant_id is not a plain name")
        if req.sample_size is not None and not 1 <= req.sample_size <= MAX_SAMPLE:
            raise RelabelRefused(422, "bad_sample_size", f"sample_size must be 1 to {MAX_SAMPLE}")
        if req.kind == KIND_PROPOSE and (req.sample_size or req.corpus_description):
            raise RelabelRefused(422, "bad_request", "propose takes no sample_size or description")

    async def _replay(self, req: JobRequest) -> Optional[TaxonomyJobRecord]:
        assert req.idempotency_key
        known = await self._repo.find_by_key(req.collection_id, req.idempotency_key)
        if known is not None:
            if not self._same(known, req):
                raise RelabelRefused(
                    409,
                    "idempotency_key_reused",
                    "this idempotency key started a job with different parameters",
                    job_id=str(known.id),
                )
            return known
        active = await self._repo.find_active(req.collection_id)
        if active is not None:
            return self._same_or_conflict(active, req)
        return None

    def _same_or_conflict(self, job: TaxonomyJobRecord, req: JobRequest) -> TaxonomyJobRecord:
        if self._same(job, req):
            return job
        raise RelabelRefused(
            409,
            "job_active",
            f"collection {req.collection_readable_id} already has a running {job.kind} job",
            job_id=str(job.id),
        )

    @staticmethod
    def _same(job: TaxonomyJobRecord, req: JobRequest) -> bool:
        """Whether ``job`` is this request, parameter for parameter.

        By the fingerprint stored with the job; a job from before fingerprints is matched on kind
        and candidate version, as it was.
        """
        stored = (job.params or {}).get("fingerprint")
        if stored is not None:
            return bool(stored == fingerprint(req))
        return (job.kind, job.candidate_version) == (req.kind, req.candidate_version)


def fingerprint(req: JobRequest) -> str:
    """A canonical hash of everything a start asks for, so a reused key is told from a replay."""
    canonical = json.dumps(
        {
            "kind": req.kind,
            "collection": str(req.collection_id),
            "tenant": req.tenant_id,
            "version": req.candidate_version,
            "sample_size": req.sample_size,
            "description": req.corpus_description,
            "cap": req.spend_cap_usd,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def job_view(job: TaxonomyJobRecord) -> Dict[str, Any]:
    """A job as the API reports it."""
    # PENDING stays its own status: a job waiting for its dispatch (or the reconciler) is not
    # running, and a client should see the delay.
    status = {
        "PENDING": "pending",
        "RUNNING": "running",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
    }.get(job.state, job.state.lower())
    return {
        "job_id": str(job.id),
        "kind": job.kind,
        "status": status,
        "state": job.state,
        "phase": job.phase,
        "collection_readable_id": job.collection_readable_id,
        "candidate_version": job.candidate_version,
        "candidate_prefix": (job.result or {}).get("candidate_prefix"),
        "result": job.result,
        "error": job.error,
        "spend_cap_usd": job.spend_cap_usd,
        "spend_used_usd": job.spend_used_usd,
        "cancel_requested": job.cancel_requested,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
