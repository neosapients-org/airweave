"""API endpoints for file downloads from storage."""

from typing import List
from zipfile import ZipFile

from fastapi import Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from airweave.api import deps
from airweave.api.context import ApiContext
from airweave.api.inject import Inject
from airweave.api.router import TrailingSlashRouter
from airweave.db.session import get_db
from airweave.domains.storage.protocols import SyncFileManagerProtocol
from airweave.models.entity import Entity

router = TrailingSlashRouter()


async def _verify_entity_ownership(
    entity_id: str,
    ctx: ApiContext,
    db: AsyncSession,
) -> None:
    """Verify the entity belongs to the authenticated user's organization.

    Looks up the entity by entity_id in Postgres and checks that
    entity.organization_id matches the JWT bearer's organization.

    Args:
        entity_id: The entity ID to verify ownership for.
        ctx: The API context (org-verified via JWT + X-Organization-ID).
        db: Database session.

    Raises:
        HTTPException: 404 if entity not found or not owned by the org.
    """
    stmt = select(Entity.id).where(
        Entity.entity_id == entity_id,
        Entity.organization_id == ctx.organization.id,
    ).limit(1)
    result = await db.execute(stmt)
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity not found: {entity_id}",
        )


@router.get("/{entity_id}", response_class=FileResponse)
async def download_file(
    *,
    entity_id: str,
    ctx: ApiContext = Depends(deps.get_context),
    db: AsyncSession = Depends(get_db),
    sfm: SyncFileManagerProtocol = Inject(SyncFileManagerProtocol),
) -> FileResponse:
    """Download a file by entity ID.

    Requires authentication via JWT + X-Organization-ID header.
    Verifies the entity belongs to the authenticated organization.

    Args:
        entity_id: The entity ID
        ctx: The current authentication context (org-verified via JWT)
        db: Database session for ownership verification
        sfm: Sync file manager (injected)

    Returns:
        FileResponse: The file content

    Raises:
        HTTPException: If file not found, not owned by org, or invalid entity ID
    """
    await _verify_entity_ownership(entity_id, ctx, db)

    try:
        content, file_path = await sfm.download_ctti_file(
            ctx.logger,
            entity_id,
            output_path=f"/tmp/{entity_id.replace(':', '_').replace('/', '_')}.md",
        )

        if content is None:
            raise HTTPException(
                status_code=404, detail=f"File not found for entity ID: {entity_id}"
            )

        file_suffix = entity_id.split(":")[-1] if ":" in entity_id else entity_id

        return FileResponse(
            path=file_path,
            filename=f"{file_suffix}.md",
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{file_suffix}.md"'},
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        ctx.logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/{entity_id}/content")
async def get_file_content(
    *,
    entity_id: str,
    ctx: ApiContext = Depends(deps.get_context),
    db: AsyncSession = Depends(get_db),
    sfm: SyncFileManagerProtocol = Inject(SyncFileManagerProtocol),
) -> dict:
    """Get file content as JSON response.

    Requires authentication via JWT + X-Organization-ID header.
    Verifies the entity belongs to the authenticated organization.

    Args:
        entity_id: The entity ID
        ctx: The current authentication context (org-verified via JWT)
        db: Database session for ownership verification
        sfm: Sync file manager (injected)

    Returns:
        dict: JSON response with the file content

    Raises:
        HTTPException: If file not found, not owned by org, or invalid entity ID
    """
    await _verify_entity_ownership(entity_id, ctx, db)

    try:
        content = await sfm.get_ctti_file_content(ctx.logger, entity_id)

        if content is None:
            raise HTTPException(
                status_code=404, detail=f"File not found for entity ID: {entity_id}"
            )

        id_suffix = entity_id.split(":")[-1] if ":" in entity_id else entity_id

        return {
            "entity_id": entity_id,
            "id": id_suffix,
            "content": content,
            "content_length": len(content),
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        ctx.logger.error(f"Error getting file content: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/batch-download", response_class=StreamingResponse)
async def download_files_batch(
    *,
    entity_ids: List[str],
    ctx: ApiContext = Depends(deps.get_context),
    db: AsyncSession = Depends(get_db),
    sfm: SyncFileManagerProtocol = Inject(SyncFileManagerProtocol),
) -> StreamingResponse:
    """Download multiple files as a ZIP archive.

    Requires authentication via JWT + X-Organization-ID header.
    Verifies all entities belong to the authenticated organization.

    Args:
        entity_ids: List of entity IDs to download
        ctx: The current authentication context (org-verified via JWT)
        db: Database session for ownership verification
        sfm: Sync file manager (injected)

    Returns:
        StreamingResponse: ZIP file containing all requested files

    Raises:
        HTTPException: If no valid files found or entities not owned by org
    """
    if not entity_ids:
        raise HTTPException(status_code=400, detail="No entity IDs provided")

    if len(entity_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files can be downloaded at once")

    # Batch ownership check: verify all entity_ids belong to this org
    stmt = select(Entity.entity_id).where(
        Entity.entity_id.in_(entity_ids),
        Entity.organization_id == ctx.organization.id,
    )
    result = await db.execute(stmt)
    owned_ids = {row[0] for row in result.all()}
    unauthorized_ids = set(entity_ids) - owned_ids
    if unauthorized_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Entities not found: {', '.join(list(unauthorized_ids)[:5])}",
        )

    try:
        results = await sfm.download_ctti_files_batch(
            ctx.logger, entity_ids, continue_on_error=True
        )

        successful_downloads = {
            entity_id: content for entity_id, (content, _) in results.items() if content is not None
        }

        if not successful_downloads:
            raise HTTPException(
                status_code=404, detail="No valid files found for the provided entity IDs"
            )

        import io

        zip_buffer = io.BytesIO()

        with ZipFile(zip_buffer, "w") as zip_file:
            for entity_id, content in successful_downloads.items():
                file_suffix = entity_id.split(":")[-1] if ":" in entity_id else entity_id
                zip_file.writestr(f"{file_suffix}.md", content)

        zip_buffer.seek(0)

        ctx.logger.info(
            f"Batch download completed: {len(successful_downloads)}/{len(entity_ids)} files",
            extra={
                "requested": len(entity_ids),
                "successful": len(successful_downloads),
                "failed": len(entity_ids) - len(successful_downloads),
            },
        )

        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="files_{len(successful_downloads)}.zip"'
                )
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        ctx.logger.error(f"Error in batch download: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/", response_model=dict)
async def check_files_exist(
    *,
    entity_ids: List[str] = Query(..., description="List of entity IDs to check"),
    ctx: ApiContext = Depends(deps.get_context),
    db: AsyncSession = Depends(get_db),
    sfm: SyncFileManagerProtocol = Inject(SyncFileManagerProtocol),
) -> dict:
    """Check which files exist in storage.

    Requires authentication via JWT + X-Organization-ID header.
    Only checks entities owned by the authenticated organization.

    Args:
        entity_ids: List of entity IDs to check
        ctx: The current authentication context (org-verified via JWT)
        db: Database session for ownership verification
        sfm: Sync file manager (injected)

    Returns:
        dict: Dictionary with entity_ids as keys and existence status as values
    """
    if not entity_ids:
        raise HTTPException(status_code=400, detail="No entity IDs provided")

    if len(entity_ids) > 1000:
        raise HTTPException(
            status_code=400, detail="Maximum 1000 entity IDs can be checked at once"
        )

    # Only check entities that belong to this org
    stmt = select(Entity.entity_id).where(
        Entity.entity_id.in_(entity_ids),
        Entity.organization_id == ctx.organization.id,
    )
    result = await db.execute(stmt)
    owned_ids = {row[0] for row in result.all()}

    results = {}

    for entity_id in entity_ids:
        if entity_id not in owned_ids:
            # Entity not owned by this org — report as not found
            results[entity_id] = False
            continue
        try:
            exists = await sfm.check_ctti_file_exists(ctx.logger, entity_id)
            results[entity_id] = exists
        except Exception as e:
            ctx.logger.warning(f"Error checking file {entity_id}: {e}")
            results[entity_id] = False

    return {
        "results": results,
        "total": len(entity_ids),
        "found": sum(1 for exists in results.values() if exists),
        "not_found": sum(1 for exists in results.values() if not exists),
    }
