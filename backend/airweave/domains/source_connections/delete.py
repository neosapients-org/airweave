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
from airweave.domains.syncs.protocols import SyncRepositoryProtocol, SyncServiceProtocol
from airweave.schemas.source_connection import SourceConnection as SourceConnectionSchema


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
    ) -> None:
        self._sc_repo = sc_repo
        self._collection_repo = collection_repo
        self._response_builder = response_builder
        self._sync_service = sync_service
        self._sync_repo = sync_repo

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

        sync_id = source_conn.sync_id
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

        return response
