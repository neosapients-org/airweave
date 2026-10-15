"""Protocols for entity action handlers."""

from typing import TYPE_CHECKING, Any, List, Protocol, Set, runtime_checkable

if TYPE_CHECKING:
    from airweave.domains.sync_pipeline.contexts import SyncContext
    from airweave.domains.sync_pipeline.contexts.runtime import SyncRuntime
    from airweave.domains.sync_pipeline.entity.actions import (
        EntityActionBatch,
        EntityDeleteAction,
        EntityInsertAction,
        EntityUpdateAction,
    )


@runtime_checkable
class EntityActionHandler(Protocol):
    """Protocol for entity action handlers.

    Handlers receive resolved entity actions and persist them to their destination.

    Contract:
    - Handlers MUST be idempotent (safe to retry on failure)
    - Handlers MUST raise SyncFailureError for non-recoverable errors
    """

    @property
    def name(self) -> str:
        """Handler name for logging and debugging."""
        ...

    async def handle_batch(
        self,
        batch: "EntityActionBatch",
        sync_context: "SyncContext",
        runtime: "SyncRuntime",
    ) -> Set[str]:
        """Handle a full action batch (main entry point).

        Args:
            batch: Entity action batch
            sync_context: Sync context
            runtime: Sync runtime with entity_tracker, source, etc.

        Returns:
            entity_ids from ``batch`` that this handler did NOT actually persist
            (e.g. conversion/embedding produced no content). The dispatcher
            excludes these from what it hands to the metadata handler, so a
            handler that silently drops an entity must report it here — an
            entity marked synced in Postgres with nothing behind it in the
            destination is invisible and permanently unretried. Handlers with
            no such silent-drop path (e.g. raw/metadata stores that raise on
            any failure) simply return an empty set.

        Raises:
            SyncFailureError: If any operation fails
        """
        ...

    async def handle_inserts(
        self,
        actions: List["EntityInsertAction"],
        sync_context: "SyncContext",
        runtime: "SyncRuntime",
    ) -> Any:
        """Handle insert actions."""
        ...

    async def handle_updates(
        self,
        actions: List["EntityUpdateAction"],
        sync_context: "SyncContext",
        runtime: "SyncRuntime",
    ) -> Any:
        """Handle update actions."""
        ...

    async def handle_deletes(
        self,
        actions: List["EntityDeleteAction"],
        sync_context: "SyncContext",
    ) -> Any:
        """Handle delete actions."""
        ...

    async def handle_orphan_cleanup(
        self,
        orphan_ids: List[str],
        sync_context: "SyncContext",
    ) -> Any:
        """Handle orphaned entity cleanup at sync end."""
        ...
