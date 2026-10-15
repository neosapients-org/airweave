"""Source that reads files uploaded through the neo_file_upload API endpoint.

Unlike every other source, this one does not reach out to a third-party
system: a dedicated upload endpoint (``api/v1/endpoints/neo_file_uploads.py``)
writes files directly into the storage backend under a per-collection prefix,
and this source lists and reads them back on each sync.

Files persist in storage until the source connection is deleted (see the
delete hook in ``domains/source_connections/delete.py``), so a normal or
incremental sync can run any number of times without losing previously
indexed documents — there is nothing analogous to "the API went away" the
way there would be for a third-party source.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from airweave.core.logging import ContextualLogger
from airweave.domains.browse_tree.types import NodeSelectionData
from airweave.domains.sources.token_providers.protocol import SourceAuthProvider
from airweave.domains.storage.exceptions import FileSkippedException, StorageException
from airweave.domains.storage.file_service import FileService
from airweave.domains.storage.protocols import StorageBackend
from airweave.domains.syncs.cursors.cursor import SyncCursor
from airweave.platform.configs.auth import NeoFileUploadAuthConfig
from airweave.platform.configs.config import NeoFileUploadConfig
from airweave.platform.decorators import source
from airweave.platform.entities._base import BaseEntity, Breadcrumb
from airweave.platform.entities.neo_file_upload import NeoUploadedFileEntity
from airweave.platform.http_client.airweave_client import AirweaveHttpClient
from airweave.platform.sources._base import BaseSource
from airweave.schemas.source_connection import AuthenticationMethod


@source(
    name="File Upload",
    short_name="neo_file_upload",
    auth_methods=[AuthenticationMethod.DIRECT],
    oauth_type=None,
    auth_config_class=NeoFileUploadAuthConfig,
    config_class=NeoFileUploadConfig,
    labels=["Files"],
    supports_continuous=False,
)
class NeoFileUploadSource(BaseSource):
    """Reads files previously written to storage by the upload API endpoint."""

    @property
    def storage(self) -> StorageBackend:
        """Get the storage backend from the DI container.

        Sources are not constructed with the container's dependencies, so we
        reach into the global container the same way ``SnapshotSource`` does
        — see the "[code blue] todo" note there for the intended future fix.
        """
        if self._storage is None:
            from airweave.core import container as container_mod

            self._storage = container_mod.container.storage_backend
        return self._storage

    @classmethod
    async def create(
        cls,
        *,
        auth: SourceAuthProvider,
        logger: ContextualLogger,
        http_client: AirweaveHttpClient,
        config: NeoFileUploadConfig,
    ) -> "NeoFileUploadSource":
        """Create a configured NeoFileUploadSource from its config."""
        instance = cls(auth=auth, logger=logger, http_client=http_client)
        instance.upload_prefix = config.upload_prefix.rstrip("/")
        instance._storage = None
        return instance

    async def generate_entities(
        self,
        *,
        cursor: SyncCursor | None = None,
        files: FileService | None = None,
        node_selections: list[NodeSelectionData] | None = None,
    ) -> AsyncGenerator[BaseEntity, None]:
        """Yield one entity per file stored under this connection's upload prefix."""
        assert files is not None, "FileService is required for neo_file_upload"

        stored_paths = await self.storage.list_files(self.upload_prefix)
        self.logger.info(
            f"neo_file_upload: found {len(stored_paths)} file(s) under {self.upload_prefix}"
        )

        for full_path in stored_paths:
            relative_path = full_path
            if relative_path.startswith(self.upload_prefix + "/"):
                relative_path = relative_path[len(self.upload_prefix) + 1 :]

            entity = await self._build_entity(relative_path, full_path, files)
            if entity is not None:
                yield entity

    async def _build_entity(
        self,
        relative_path: str,
        full_path: str,
        files: FileService,
    ) -> Optional[NeoUploadedFileEntity]:
        """Read one stored file and turn it into an entity, or skip it."""
        segments = relative_path.split("/")
        file_name = segments[-1]
        folder_segments = segments[:-1]

        try:
            content = await self.storage.read_file(full_path)
        except StorageException as e:
            self.logger.warning(f"neo_file_upload: failed to read {full_path}: {e}")
            return None

        breadcrumbs = [
            Breadcrumb(entity_id=seg, name=seg, entity_type="NeoUploadFolder")
            for seg in folder_segments
        ]

        entity = NeoUploadedFileEntity(
            id=relative_path,
            name=file_name,
            relative_path=relative_path,
            folder_segments=folder_segments,
            uploaded_at=datetime.now(timezone.utc),
            breadcrumbs=breadcrumbs,
            url=f"neo-file-upload://{relative_path}",
            size=len(content),
            file_type="document",
        )

        try:
            await files.save_bytes(
                entity=entity,
                content=content,
                filename_with_extension=file_name,
                logger=self.logger,
            )
        except FileSkippedException as e:
            self.logger.info(f"neo_file_upload: skipping {relative_path}: {e.reason}")
            return None

        return entity

    async def validate(self) -> None:
        """Valid as long as the storage backend is reachable — no third party to check."""
        await self.storage.exists(self.upload_prefix)
