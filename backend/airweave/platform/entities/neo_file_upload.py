"""Entity schema for the neo_file_upload source."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from airweave.platform.entities._airweave_field import AirweaveField
from airweave.platform.entities._base import FileEntity


class NeoUploadedFileEntity(FileEntity):
    """Schema for a file uploaded through the neo_file_upload API endpoint.

    The entity id is the file's storage-relative path, so re-uploading the
    same path is treated as an update rather than a new file, and content
    hashing (done upstream on the raw bytes) decides whether it needs to be
    re-embedded.
    """

    id: str = AirweaveField(
        ...,
        description="Storage-relative path of the uploaded file, used as the entity id",
        is_entity_id=True,
    )
    name: str = AirweaveField(
        ...,
        description="File name (last path segment)",
        is_name=True,
        embeddable=True,
    )
    relative_path: str = AirweaveField(
        ...,
        description="Full relative path within the upload, including folder segments",
        embeddable=True,
    )
    folder_segments: List[str] = AirweaveField(
        default_factory=list,
        description="Folder path segments the file was uploaded under, in order",
        embeddable=False,
    )
    uploaded_at: Optional[datetime] = AirweaveField(
        None,
        # Must stay a real `datetime`, not an ISO string — the entity
        # pipeline copies this value verbatim into `BaseEntity.created_at`
        # via plain attribute assignment (no Pydantic re-validation/coercion
        # on that path), and the Vespa transformer later calls
        # `.timestamp()` on it. A string here fails that call for every
        # single chunk, silently (`dest_build` logs "Failed to transform
        # entity" per chunk but the sync still reports success), so nothing
        # ever reaches Vespa despite `SyncStats` claiming N inserted.
        description="Timestamp of when the file was uploaded",
        embeddable=False,
        is_created_at=True,
    )
