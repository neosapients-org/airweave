"""Source connection deletion service."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from airweave import schemas
from airweave.api.context import ApiContext
from airweave.core.exceptions import NotFoundException
from airweave.domains.collections.protocols import CollectionRepositoryProtocol
from airweave.domains.source_connections.protocols import (
    ResponseBuilderProtocol,
    SourceConnectionDeletionServiceProtocol,
    SourceConnectionRepositoryProtocol,
)
from airweave.domains.storage.paths import paths as storage_paths
from airweave.domains.storage.protocols import StorageBackend
from airweave.domains.syncs.protocols import SyncRepositoryProtocol, SyncServiceProtocol
from airweave.schemas.source_connection import SourceConnection as SourceConnectionSchema

# Source short_names whose entities are backed by files this service owns in
# the storage backend (rather than data pulled live from a third party), so
# deleting the connection must also delete those files or they leak forever.
_SOURCES_WITH_OWNED_STORAGE_FILES = frozenset({"neo_file_upload"})


class SourceConnectionDeletionService(SourceConnectionDeletionServiceProtocol):
    """Deletes a source connection and all related data.

    The flow is:
    1. Delegate cancel + wait + Vespa purge + cleanup scheduling to SyncService.delete.
    2. Remove the source connection row.
    3. Remove the sync row, which CASCADE-deletes its entities, jobs, entity
       counts, cursor, and sync connections via the ``ondelete="CASCADE"`` FKs.
    """

    def __init__(  # noqa: D107
        self,
        sc_repo: SourceConnectionRepositoryProtocol,
        collection_repo: CollectionRepositoryProtocol,
        response_builder: ResponseBuilderProtocol,
        sync_service: SyncServiceProtocol,
        sync_repo: SyncRepositoryProtocol,
        storage_backend: StorageBackend,
    ) -> None:
        self._sc_repo = sc_repo
        self._collection_repo = collection_repo
        self._response_builder = response_builder
        self._sync_service = sync_service
        self._sync_repo = sync_repo
        self._storage = storage_backend

    async def delete(
        self,
        db: AsyncSession,
        *,
        id: UUID,
        ctx: ApiContext,
    ) -> SourceConnectionSchema:
        """Delete a source connection and all related data."""
        source_conn = await self._sc_repo.get(db, id=id, ctx=ctx)
        if not source_conn:
            raise NotFoundException("Source connection not found")

        # Read everything needed after removal up front: removing the row expires
        # ``source_conn``, and touching an expired attribute on an AsyncSession
        # triggers a sync lazy refresh that raises ``MissingGreenlet``.
        sync_id = source_conn.sync_id
        owns_storage_files = source_conn.short_name in _SOURCES_WITH_OWNED_STORAGE_FILES
        collection_orm = await self._collection_repo.get_by_readable_id(
            db, readable_id=source_conn.readable_collection_id, ctx=ctx
        )
        if not collection_orm:
            raise NotFoundException("Collection not found")
        collection = schemas.CollectionRecord.model_validate(collection_orm, from_attributes=True)

        response = await self._response_builder.build_response(db, source_conn, ctx)

        if sync_id:
            await self._sync_service.delete(
                db,
                sync_id=sync_id,
                collection_id=collection.id,
                organization_id=collection.organization_id,
                ctx=ctx,
                cancel_timeout_seconds=15,
            )
            # Removing the sync row hard-deletes everything hanging off it via the
            # ``ondelete="CASCADE"`` FKs: the source connection (its ``sync_id`` FK
            # cascades), entities, sync jobs, entity counts, cursor, and sync
            # connections. This is what prevents orphaned entity/job rows.
            await self._sync_repo.remove(db, id=sync_id, ctx=ctx)
        else:
            # No sync attached (e.g. an unauthenticated connection): just remove
            # the source connection row directly.
            await self._sc_repo.remove(db, id=id, ctx=ctx)

        if owns_storage_files:
            await self._cleanup_owned_storage_files(collection, ctx)

        return response

    async def _cleanup_owned_storage_files(
        self, collection: schemas.CollectionRecord, ctx: ApiContext
    ) -> None:
        """Best-effort delete of files this connection owns in storage.

        Failure here must not fail the connection delete — the connection
        and sync rows are already gone — but a leftover file costs storage
        forever with no other cleanup path, so it is logged loudly.
        """
        upload_prefix = storage_paths.upload_prefix(ctx.organization.id, collection.readable_id)
        try:
            deleted = await self._storage.delete(upload_prefix)
            ctx.logger.info(
                f"Deleted uploaded files at {upload_prefix} (found_any={deleted})"
            )
        except Exception as e:
            ctx.logger.warning(
                f"Failed to delete uploaded files at {upload_prefix}: {e}. "
                "These files are now orphaned in storage and need manual cleanup."
            )
