"""Tests for the `browse_collection` endpoint handler.

The handler itself is thin: feature-flag gate -> usage check -> delegate to
BrowseService. We call it directly (bypassing FastAPI DI) and verify each
branch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from airweave.api.context import ApiContext
from airweave.api.v1.endpoints.search import browse_collection
from airweave.core.logging import logger
from airweave.core.shared_models import AuthMethod, FeatureFlag
from airweave.domains.search.fakes.browse import FakeBrowseService
from airweave.domains.usage.types import ActionType
from airweave.schemas.search_v2 import BrowseRequest, BrowseResponse


def _make_ctx(features: list[FeatureFlag] | None = None) -> ApiContext:
    from airweave.schemas.organization import Organization

    now = datetime.now(timezone.utc)
    org = Organization(
        id=str(uuid4()),
        name="Test Org",
        created_at=now,
        modified_at=now,
        enabled_features=features or [],
    )
    return ApiContext(
        request_id="test-req-001",
        organization=org,
        auth_method=AuthMethod.SYSTEM,
        auth_metadata={},
        logger=logger.with_context(request_id="test-req-001"),
    )


class TestBrowseCollectionEndpoint:
    """Direct tests for the browse_collection handler."""

    @pytest.mark.asyncio
    async def test_returns_404_when_feature_flag_disabled(self) -> None:
        ctx = _make_ctx(features=[])
        usage_checker = AsyncMock()
        service = FakeBrowseService()

        with pytest.raises(HTTPException) as exc_info:
            await browse_collection(
                readable_id="any",
                request=BrowseRequest(),
                db=AsyncMock(),
                ctx=ctx,
                usage_checker=usage_checker,
                service=service,
            )

        assert exc_info.value.status_code == 404
        # Service should not be called when flag is off.
        assert service._calls == []
        usage_checker.is_allowed.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_usage_checker_and_service_when_enabled(self) -> None:
        ctx = _make_ctx(features=[FeatureFlag.COLLECTION_BROWSE])
        usage_checker = AsyncMock()
        service = FakeBrowseService()
        service.seed_response(
            BrowseResponse(results=[], total=7, limit=50, offset=0)
        )

        response = await browse_collection(
            readable_id="my-col",
            request=BrowseRequest(limit=50),
            db=AsyncMock(),
            ctx=ctx,
            usage_checker=usage_checker,
            service=service,
        )

        assert response.total == 7
        # Usage check fired with the right args.
        usage_checker.is_allowed.assert_awaited_once()
        args = usage_checker.is_allowed.await_args.args
        # signature: (db, org_id, action_type)
        assert args[1] == ctx.organization.id
        assert args[2] == ActionType.QUERIES
        # Service was called with the readable_id.
        assert len(service._calls) == 1
        assert service._calls[0][1] == "my-col"
