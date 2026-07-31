"""In-memory fake for BrowseServiceProtocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from airweave.api.context import ApiContext
from airweave.domains.search.protocols import BrowseServiceProtocol
from airweave.schemas.search_v2 import BrowseResponse

if TYPE_CHECKING:
    from airweave.schemas.search_v2 import BrowseRequest


class FakeBrowseService(BrowseServiceProtocol):
    """Returns a seeded BrowseResponse. Records calls."""

    def __init__(self) -> None:
        self._response: BrowseResponse = BrowseResponse(results=[], total=0, limit=50, offset=0)
        self._calls: list[tuple] = []

    def seed_response(self, response: BrowseResponse) -> None:
        self._response = response

    async def browse(
        self,
        db: AsyncSession,
        ctx: ApiContext,
        readable_id: str,
        request: BrowseRequest,
    ) -> BrowseResponse:
        self._calls.append(("browse", readable_id, request))
        return self._response
