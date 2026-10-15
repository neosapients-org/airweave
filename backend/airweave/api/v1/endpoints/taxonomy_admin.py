"""Admin endpoints for taxonomy re-labelling (contract 7; P7 design sections 5 and 12).

Same permission check as ``/admin/collections/{id}/reclassify``: an admin user, or an API key
whose organization has the ``API_KEY_ADMIN_SYNC`` feature. Every route is scoped to the caller's
organization: another organization's collection or job is answered as not found.
"""

from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import Body, Depends, Header, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from airweave.api import deps
from airweave.api.context import ApiContext
from airweave.api.deps import Inject
from airweave.api.router import TrailingSlashRouter
from airweave.api.v1.endpoints.admin import _require_admin_permission
from airweave.core import container as container_mod
from airweave.core.shared_models import FeatureFlag as FeatureFlagEnum
from airweave.domains.collections.protocols import CollectionRepositoryProtocol
from airweave.domains.taxonomy.relabel.jobs import (
    CANCELLED,
    COMPLETED,
    FAILED,
    MODE_FULL,
    SCOPE_DIFFERING,
    RelabelJob,
)
from airweave.domains.taxonomy.relabel.service import (
    RelabelRefused,
    RelabelService,
    StartRequest,
)

router = TrailingSlashRouter()

_STATUS = {COMPLETED: "completed", FAILED: "failed", CANCELLED: "cancelled"}


class RelabelRequest(BaseModel):
    """Body of ``POST /admin/collections/{readable_id}/relabel``."""

    model_config = ConfigDict(extra="forbid")

    taxonomy_version: str = Field(..., description="The version to label with (the active file's)")
    mode: str = Field(
        MODE_FULL, description="`full`, or `incremental` (the target only adds categories)"
    )
    scope: str = Field(
        SCOPE_DIFFERING,
        description="`differing` (chunks at another version) or `not_labelled` (bootstrap)",
    )
    idempotency_key: Optional[str] = Field(None, description="A replay returns the same job")
    spend_cap_usd: Optional[float] = Field(
        None, description="Lowers the approved ceiling for this job; never raises it"
    )


def get_relabel_service() -> RelabelService:
    """The service, or 503 while the taxonomy step is switched off."""
    service: Optional[RelabelService] = getattr(container_mod.container, "relabel_service", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="Taxonomy labelling is not enabled (TAXONOMY_ENABLED)"
        )
    return service


def job_view(job: RelabelJob) -> Dict[str, Any]:
    """A job as the API reports it: the contract's fields first, then the details."""
    status = _STATUS.get(job.state, "running")
    return {
        "relabel_job_id": str(job.id),
        "status": status,
        "state": job.state,
        "chunks_relabelled": job.relabelled_in_total,
        "chunks_total": job.chunks_total,
        "error": job.error,
        "collection_readable_id": job.collection_readable_id,
        "taxonomy_version": job.target_version,
        "previous_taxonomy_version": job.previous_version,
        "mode": job.mode,
        "scope": job.scope,
        "chunks_classified": job.chunks_relabelled,
        "chunks_stamped": job.chunks_stamped,
        "chunks_skipped_changed": job.chunks_skipped_changed,
        "chunks_unlabellable": job.chunks_unlabellable,
        "chunks_failed": job.chunks_failed,
        "spend_estimate_usd": job.spend_estimate_usd,
        "spend_cap_usd": job.spend_cap_usd,
        "cancel_requested": job.cancel_requested,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


async def _collection(
    db: AsyncSession, repo: CollectionRepositoryProtocol, readable_id: str, ctx: ApiContext
) -> Any:
    collection = await repo.get_by_readable_id(db, readable_id=readable_id, ctx=ctx)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


@router.post("/collections/{readable_id}/relabel", status_code=202)
async def start_relabel(
    readable_id: str = Path(..., description="The readable ID of the collection"),
    body: RelabelRequest = Body(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(deps.get_db),
    ctx: ApiContext = Depends(deps.get_context),
    collection_repo: CollectionRepositoryProtocol = Inject(CollectionRepositoryProtocol),
    service: RelabelService = Depends(get_relabel_service),
) -> Dict[str, Any]:
    """Admin-only: start a re-label job (202), or return the one a replay refers to."""
    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    collection = await _collection(db, collection_repo, readable_id, ctx)
    request = StartRequest(
        organization_id=ctx.organization.id,
        collection_id=collection.id,
        collection_readable_id=collection.readable_id,
        taxonomy_version=body.taxonomy_version,
        mode=body.mode,
        scope=body.scope,
        idempotency_key=body.idempotency_key or idempotency_key,
        spend_cap_usd=body.spend_cap_usd,
    )
    try:
        job, created = await service.start(request)
    except RelabelRefused as e:
        raise HTTPException(status_code=e.status, detail=e.detail()) from e
    ctx.logger.info(
        f"Re-label job {job.id} for {readable_id} -> {job.target_version} "
        f"({'accepted' if created else 'replay'}, mode={job.mode}, scope={job.scope}, "
        f"estimate=USD {job.spend_estimate_usd}, cap=USD {job.spend_cap_usd})"
    )
    return {**job_view(job), "replayed": not created}


@router.get("/collections/{readable_id}/taxonomy-statistics")
async def taxonomy_statistics(
    readable_id: str = Path(..., description="The readable ID of the collection"),
    target_version: str = Query("", description="The version to compare with; may be empty"),
    db: AsyncSession = Depends(deps.get_db),
    ctx: ApiContext = Depends(deps.get_context),
    collection_repo: CollectionRepositoryProtocol = Inject(CollectionRepositoryProtocol),
    service: RelabelService = Depends(get_relabel_service),
) -> Dict[str, Any]:
    """Admin-only: chunk counts for a collection against a target taxonomy version.

    ``collectionId`` is Airweave's own id of the collection, which is what its chunks carry.
    Read-only: one scroll over the collection's chunks, small fields only.
    """
    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    collection = await _collection(db, collection_repo, readable_id, ctx)
    try:
        stats = await service.statistics(collection.id, target_version)
    except RelabelRefused as e:
        raise HTTPException(status_code=e.status, detail=e.detail()) from e
    except Exception as e:
        ctx.logger.error(f"Taxonomy statistics for {readable_id} failed: {e}")
        raise HTTPException(status_code=502, detail="Vespa could not be read") from e
    return stats.to_port()


@router.get("/collections/{readable_id}/unresolved")
async def list_unresolved(
    readable_id: str = Path(..., description="The readable ID of the collection"),
    limit: int = Query(100, description="Target page size, 1 to 500; a page may return a few more"),
    cursor: Optional[str] = Query(None, description="`next_cursor` of the previous page"),
    taxonomy_version: Optional[str] = Query(
        None, description="Only chunks labelled by this version"
    ),
    db: AsyncSession = Depends(deps.get_db),
    ctx: ApiContext = Depends(deps.get_context),
    collection_repo: CollectionRepositoryProtocol = Inject(CollectionRepositoryProtocol),
    service: RelabelService = Depends(get_relabel_service),
) -> Dict[str, Any]:
    """Admin-only, read-only: scroll the chunks whose status is UNRESOLVED or PARTIALLY_ASSIGNED.

    These chunks are the evolution job's input (design section 6); no second copy of their text
    is stored anywhere. The text of each is the chunk's own text.
    """
    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    collection = await _collection(db, collection_repo, readable_id, ctx)
    try:
        chunks, next_cursor = await service.unresolved(
            collection.id, limit, cursor, taxonomy_version
        )
    except RelabelRefused as e:
        raise HTTPException(status_code=e.status, detail=e.detail()) from e
    except Exception as e:
        ctx.logger.error(f"Unresolved scroll for {readable_id} failed: {e}")
        raise HTTPException(status_code=502, detail="Vespa could not be read") from e
    return {"chunks": chunks, "next_cursor": next_cursor}


@router.get("/relabel-jobs/{relabel_job_id}")
async def get_relabel_job(
    relabel_job_id: UUID = Path(...),
    ctx: ApiContext = Depends(deps.get_context),
    service: RelabelService = Depends(get_relabel_service),
) -> Dict[str, Any]:
    """Admin-only: a re-label job's state and counts."""
    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    job = await service.get(relabel_job_id, ctx.organization.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Re-label job not found")
    return job_view(job)


@router.post("/relabel-jobs/{relabel_job_id}/cancel")
async def cancel_relabel_job(
    relabel_job_id: UUID = Path(...),
    ctx: ApiContext = Depends(deps.get_context),
    service: RelabelService = Depends(get_relabel_service),
) -> Dict[str, Any]:
    """Admin-only: stop a job. A running job stops at its next page; an unclaimed one at once."""
    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    job = await service.cancel(relabel_job_id, ctx.organization.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Re-label job not found")
    return job_view(job)


# --------------------------------------------------------------------------- discover and propose


class DiscoveryRequest(BaseModel):
    """Body of ``POST /admin/collections/{readable_id}/taxonomy-discovery``."""

    model_config = ConfigDict(extra="forbid")

    candidate_version: str = Field(..., description="Name of the candidate folder and version")
    idempotency_key: str = Field(..., description="A replay returns the same job")
    tenant_id: str = Field(
        ...,
        description="The tenant whose prefix receives the candidate (a first version has no "
        "active file to read it from)",
    )
    spend_cap_usd: Optional[float] = Field(None, description="Lowers the approved ceiling")
    sample_size: Optional[int] = Field(None, description="Documents to sample")
    corpus_description: Optional[str] = Field(
        None, max_length=500, description="Inserted into the reviewer's prompts"
    )


class ProposalRequest(BaseModel):
    """Body of ``POST /admin/collections/{readable_id}/taxonomy-evolution/propose``."""

    model_config = ConfigDict(extra="forbid")

    candidate_version: str
    idempotency_key: str
    spend_cap_usd: Optional[float] = None


def get_job_service() -> Any:
    """The discover/propose service, or 503 while the jobs are off."""
    service = getattr(container_mod.container, "taxonomy_job_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Taxonomy discover and propose jobs are not enabled (TAXONOMY_JOBS_ENABLED)",
        )
    return service


def _serving_tenant(container: Any) -> Any:
    processor = getattr(container, "taxonomy_processor", None)
    return processor.resolver if processor is not None else None


async def _start_job(
    kind: str,
    readable_id: str,
    body: Any,
    tenant_id: Optional[str],
    db: AsyncSession,
    ctx: ApiContext,
    collection_repo: CollectionRepositoryProtocol,
    service: Any,
) -> Dict[str, Any]:
    from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError
    from airweave.domains.taxonomy.jobs.service import JobRequest
    from airweave.domains.taxonomy.jobs.service import job_view as view

    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    collection = await _collection(db, collection_repo, readable_id, ctx)
    if tenant_id is None:  # propose: the serving version's tenant, from the active file
        resolver = _serving_tenant(container_mod.container)
        try:
            taxonomy = (
                await resolver.resolve(collection.readable_id, fresh=True) if resolver else None
            )
        except TaxonomyResolutionError as e:
            raise HTTPException(
                status_code=409, detail={"code": "taxonomy_unusable", "message": str(e)}
            ) from e
        if taxonomy is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "no_active_taxonomy",
                    "message": "the collection has no taxonomy to evolve",
                },
            )
        tenant_id = taxonomy.tenant_id
    request = JobRequest(
        kind=kind,
        organization_id=ctx.organization.id,
        collection_id=collection.id,
        collection_readable_id=collection.readable_id,
        tenant_id=tenant_id,
        candidate_version=body.candidate_version,
        idempotency_key=body.idempotency_key,
        spend_cap_usd=body.spend_cap_usd,
        sample_size=getattr(body, "sample_size", None),
        corpus_description=getattr(body, "corpus_description", None),
    )
    try:
        job, created = await service.start(request)
    except RelabelRefused as e:
        raise HTTPException(status_code=e.status, detail=e.detail()) from e
    ctx.logger.info(
        f"Taxonomy {kind} job {job.id} for {readable_id} -> {job.candidate_version} "
        f"({'accepted' if created else 'replay'}, cap=USD {job.spend_cap_usd})"
    )
    return {**view(job), "replayed": not created}


@router.post("/collections/{readable_id}/taxonomy-discovery", status_code=202)
async def start_taxonomy_discovery(
    readable_id: str = Path(..., description="The readable ID of the collection"),
    body: DiscoveryRequest = Body(...),
    db: AsyncSession = Depends(deps.get_db),
    ctx: ApiContext = Depends(deps.get_context),
    collection_repo: CollectionRepositoryProtocol = Inject(CollectionRepositoryProtocol),
    service: Any = Depends(get_job_service),
) -> Dict[str, Any]:
    """Admin-only: enqueue a discover job (A13). It writes a candidate first taxonomy version."""
    from airweave.domains.taxonomy.jobs.records import KIND_DISCOVER

    return await _start_job(
        KIND_DISCOVER, readable_id, body, body.tenant_id, db, ctx, collection_repo, service
    )


@router.post("/collections/{readable_id}/taxonomy-evolution/propose", status_code=202)
async def start_taxonomy_proposal(
    readable_id: str = Path(..., description="The readable ID of the collection"),
    body: ProposalRequest = Body(...),
    db: AsyncSession = Depends(deps.get_db),
    ctx: ApiContext = Depends(deps.get_context),
    collection_repo: CollectionRepositoryProtocol = Inject(CollectionRepositoryProtocol),
    service: Any = Depends(get_job_service),
) -> Dict[str, Any]:
    """Admin-only: enqueue a propose job (A12): one evolution wave on the unresolved chunks."""
    from airweave.domains.taxonomy.jobs.records import KIND_PROPOSE

    return await _start_job(
        KIND_PROPOSE, readable_id, body, None, db, ctx, collection_repo, service
    )


@router.get("/taxonomy-jobs/{job_id}")
async def get_taxonomy_job(
    job_id: UUID = Path(...),
    ctx: ApiContext = Depends(deps.get_context),
    service: Any = Depends(get_job_service),
) -> Dict[str, Any]:
    """Admin-only: a discover or propose job's state, result and cost."""
    from airweave.domains.taxonomy.jobs.service import job_view as view

    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    job = await service.get(job_id, ctx.organization.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Taxonomy job not found")
    return view(job)


@router.post("/taxonomy-jobs/{job_id}/cancel")
async def cancel_taxonomy_job(
    job_id: UUID = Path(...),
    ctx: ApiContext = Depends(deps.get_context),
    service: Any = Depends(get_job_service),
) -> Dict[str, Any]:
    """Admin-only: stop a job. Nothing is written to S3 after the cancel is seen."""
    from airweave.domains.taxonomy.jobs.service import job_view as view

    _require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)
    job = await service.cancel(job_id, ctx.organization.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Taxonomy job not found")
    return view(job)
