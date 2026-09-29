"""API endpoints for uploading files that back the neo_file_upload source.

Files posted here are written straight into the storage backend under a
per-collection prefix (``StoragePaths.upload_prefix``). They are not synced
until a caller separately creates or runs a ``neo_file_upload`` source
connection for the collection — this endpoint only stores bytes.
"""

from __future__ import annotations

from typing import List

from fastapi import Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from airweave.api import deps
from airweave.api.context import ApiContext
from airweave.api.inject import Inject
from airweave.api.router import TrailingSlashRouter
from airweave.domains.collections.exceptions import CollectionNotFoundError
from airweave.domains.collections.protocols import CollectionServiceProtocol
from airweave.domains.storage.paths import paths
from airweave.domains.storage.protocols import StorageBackend
from airweave.domains.sync_pipeline.file_types import SUPPORTED_FILE_EXTENSIONS

router = TrailingSlashRouter()

# Conservative defaults; this is a neo-platform-specific surface, not a
# generic Airweave setting, so the limits live here rather than in Settings.
MAX_FILES_PER_UPLOAD = 200
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_TOTAL_UPLOAD_BYTES = 150 * 1024 * 1024  # 150 MB


class NeoFileUploadResult(BaseModel):
    """Result of one uploaded-files batch."""

    stored: List[str]
    rejected: List[dict]


def _extension_of(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


@router.post("/{readable_collection_id}", response_model=NeoFileUploadResult)
async def upload_files(
    *,
    readable_collection_id: str,
    files: List[UploadFile] = File(...),
    paths_: List[str] = Form(default=[], alias="paths"),
    ctx: ApiContext = Depends(deps.get_context),
    db: AsyncSession = Depends(deps.get_db),
    collection_service: CollectionServiceProtocol = Inject(CollectionServiceProtocol),
    storage: StorageBackend = Inject(StorageBackend),
) -> NeoFileUploadResult:
    """Store uploaded files under a collection's upload prefix.

    ``readable_collection_id`` must belong to the caller's organization —
    ``CollectionService.get`` enforces that and this endpoint 404s otherwise,
    the same response a caller gets for a collection that doesn't exist at
    all, so this can't be used to probe for other orgs' collection ids.

    Each file's relative path (used to preserve uploaded folder structure)
    comes from the parallel ``paths`` form field when provided, falling back
    to the file's own filename. A rejected file is reported in ``rejected``
    rather than failing the whole batch.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)} (max {MAX_FILES_PER_UPLOAD})",
        )

    try:
        collection = await collection_service.get(
            db, readable_id=readable_collection_id, ctx=ctx
        )
    except CollectionNotFoundError as e:
        raise HTTPException(status_code=404, detail="Collection not found") from e

    upload_prefix = paths.upload_prefix(ctx.organization.id, collection.readable_id)

    relative_paths = paths_ or [f.filename or "" for f in files]
    if len(relative_paths) != len(files):
        raise HTTPException(
            status_code=400, detail="paths[] must have the same length as files[]"
        )

    stored: List[str] = []
    rejected: List[dict] = []
    total_bytes = 0

    for upload, raw_relative_path in zip(files, relative_paths):
        safe_relative_path = paths.safe_relative_path(raw_relative_path or upload.filename or "")
        if safe_relative_path is None:
            rejected.append({"path": raw_relative_path, "reason": "invalid or unsafe path"})
            continue

        ext = _extension_of(safe_relative_path)
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            rejected.append(
                {"path": safe_relative_path, "reason": f"unsupported extension: {ext or '(none)'}"}
            )
            continue

        content = await upload.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            rejected.append(
                {
                    "path": safe_relative_path,
                    "reason": f"file too large ({len(content)} bytes, max {MAX_FILE_SIZE_BYTES})",
                }
            )
            continue

        total_bytes += len(content)
        if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds total size limit of {MAX_TOTAL_UPLOAD_BYTES} bytes",
            )

        full_path = paths.upload_file_path(
            ctx.organization.id, collection.readable_id, safe_relative_path
        )
        await storage.write_file(full_path, content)
        stored.append(safe_relative_path)

    ctx.logger.info(
        "neo_file_upload: stored files",
        extra={
            "collection_readable_id": collection.readable_id,
            "stored_count": len(stored),
            "rejected_count": len(rejected),
            "upload_prefix": upload_prefix,
        },
    )

    return NeoFileUploadResult(stored=stored, rejected=rejected)
