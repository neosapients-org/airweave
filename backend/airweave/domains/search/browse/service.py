"""Browse service — paginated tabular listing of a collection.

No query, no embeddings, no ranking. Hits Vespa's filter_search() and count()
in parallel. Forces chunk_index = 0 so each source entity shows up as a single row.

Access control is search's: with a user principal (the admin ``as-user`` route) the rows and
the count carry the same ACL clause, so browse and its CSV export show only what that user's
search would. Without one, as for search, no source ACL applies. Each browse emits the search
lifecycle events (tier ``browse``) for analytics; its completion is not billed as a query.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from airweave.api.context import ApiContext
from airweave.core.events.search import SearchCompletedEvent, SearchFailedEvent, SearchTier
from airweave.core.exceptions import InvalidInputError
from airweave.core.protocols.event_bus import EventBus
from airweave.domains.access_control.protocols import AccessBrokerProtocol
from airweave.domains.collections.protocols import CollectionRepositoryProtocol
from airweave.domains.search.adapters.vector_db.protocol import VectorDBProtocol
from airweave.domains.search.executor import resolve_acl_principals
from airweave.domains.search.protocols import BrowseServiceProtocol
from airweave.domains.search.types.filters import (
    CHUNK_SCOPED_FIELDS,
    FilterableField,
    FilterCondition,
    FilterGroup,
    FilterOperator,
)
from airweave.schemas.search_v2 import BrowseResponse

if TYPE_CHECKING:
    from airweave.schemas.search_v2 import BrowseRequest


class BrowseService(BrowseServiceProtocol):
    """Browse a collection as paginated rows."""

    def __init__(
        self,
        vector_db: VectorDBProtocol,
        collection_repo: CollectionRepositoryProtocol,
        access_broker: AccessBrokerProtocol,
        event_bus: EventBus,
    ) -> None:
        """Initialize with vector DB, collection repository, access broker and event bus."""
        self._vector_db = vector_db
        self._collection_repo = collection_repo
        self._access_broker = access_broker
        self._event_bus = event_bus

    async def browse(
        self,
        db: AsyncSession,
        ctx: ApiContext,
        readable_id: str,
        request: BrowseRequest,
        user_principal_override: str | None = None,
    ) -> BrowseResponse:
        """Run a paginated unranked listing for the given collection.

        ``user_principal_override`` applies that user's source ACL, as the search tiers do.
        """
        start = time.monotonic()
        ctx.logger.info(
            f"Browse started collection={readable_id} limit={request.limit} offset={request.offset}"
        )
        try:
            response = await self._browse(db, ctx, readable_id, request, user_principal_override)
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            ctx.logger.error(
                f"Browse failed collection={readable_id} duration_ms={duration_ms} error={e}"
            )
            await self._event_bus.publish(
                SearchFailedEvent(
                    organization_id=ctx.organization.id,
                    request_id=ctx.request_id,
                    tier=SearchTier.BROWSE,
                    plan=ctx.billing_plan,
                    message=str(e),
                    duration_ms=duration_ms,
                )
            )
            raise
        duration_ms = int((time.monotonic() - start) * 1000)
        ctx.logger.info(
            f"Browse completed collection={readable_id} "
            f"rows={len(response.results)} total={response.total} duration_ms={duration_ms}"
        )
        await self._event_bus.publish(
            SearchCompletedEvent(
                organization_id=ctx.organization.id,
                request_id=ctx.request_id,
                tier=SearchTier.BROWSE,
                plan=ctx.billing_plan,
                results=[r.model_dump(mode="json") for r in response.results],
                duration_ms=duration_ms,
                # Seen by analytics; browsing a table is not billed as a query (as before).
                billable=False,
            )
        )
        return response

    async def _browse(
        self,
        db: AsyncSession,
        ctx: ApiContext,
        readable_id: str,
        request: BrowseRequest,
        user_principal_override: str | None,
    ) -> BrowseResponse:
        collection = await self._collection_repo.get_by_readable_id(db, readable_id, ctx)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection '{readable_id}' not found")

        filter_groups = self._build_filter_groups(request)
        collection_id = str(collection.id)
        # Validated at the request: None, or at least two characters once trimmed.
        name_substring = request.name_query
        acl_principals = await resolve_acl_principals(
            self._access_broker, db, ctx, user_principal_override, readable_id
        )

        page = self._vector_db.filter_search(
            filter_groups=filter_groups,
            collection_id=collection_id,
            limit=request.limit,
            offset=request.offset,
            name_substring=name_substring,
            acl_principals=acl_principals,
        )
        total: int | None = None
        if request.include_total:
            results, total = await asyncio.gather(
                page,
                self._vector_db.count(
                    filter_groups=filter_groups,
                    collection_id=collection_id,
                    name_substring=name_substring,
                    acl_principals=acl_principals,
                ),
            )
        else:
            results = await page

        return BrowseResponse(
            results=results,
            total=total,
            limit=request.limit,
            offset=request.offset,
        )

    def _build_filter_groups(self, request: BrowseRequest) -> list[FilterGroup]:
        """Combine user filters with the chunk-index=0 row anchor and convenience filters.

        chunk_index=0 collapses Vespa's per-chunk documents to one row per source entity.
        Each FilterGroup is AND-internal; multiple groups are OR'd together. To preserve
        AND semantics across user filters AND our forced conditions, we extend each user
        group's conditions instead of appending a new group.
        """
        for group in request.filter or []:
            scoped = sorted(
                {c.field.value for c in group.conditions if c.field in CHUNK_SCOPED_FIELDS}
            )
            if scoped:
                # Browse lists one row per entity, read from its first chunk: a label or page
                # that sits on a later chunk would make the entity vanish. Search matches chunks.
                raise InvalidInputError(
                    f"Browse cannot filter on chunk fields ({', '.join(scoped)}); "
                    f"use search to filter by topic, page or position."
                )

        chunk_anchor = FilterCondition(
            field=FilterableField.SYSTEM_METADATA_CHUNK_INDEX,
            operator=FilterOperator.EQUALS,
            value=0,
        )

        extra_conditions: list[FilterCondition] = [chunk_anchor]

        if request.sync_ids:
            extra_conditions.append(
                FilterCondition(
                    field=FilterableField.SYSTEM_METADATA_SYNC_ID,
                    operator=FilterOperator.IN,
                    value=list(request.sync_ids),
                )
            )

        if request.entity_types:
            extra_conditions.append(
                FilterCondition(
                    field=FilterableField.SYSTEM_METADATA_ENTITY_TYPE,
                    operator=FilterOperator.IN,
                    value=list(request.entity_types),
                )
            )

        if not request.filter:
            return [FilterGroup(conditions=extra_conditions)]

        return [
            FilterGroup(conditions=[*group.conditions, *extra_conditions])
            for group in request.filter
        ]
