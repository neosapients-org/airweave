"""Unit tests for the Microsoft Purview sensitivity-label filter.

Covers the ``SensitivityLabelFilter`` used by the SharePoint, SharePoint Online,
and OneDrive connectors to skip files and sites carrying configured Purview
label GUIDs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from airweave.platform.sources.microsoft_sensitivity_labels import SensitivityLabelFilter


@pytest.fixture
def logger():
    """Return a logger mock — the filter only needs .info/.warning."""
    mock = MagicMock()
    mock.info = MagicMock()
    mock.warning = MagicMock()
    return mock


def _response(status_code: int, json_body: Any = None) -> httpx.Response:
    """Build an httpx.Response with the given status and JSON body."""
    request = httpx.Request("POST", "https://graph.microsoft.com/v1.0/x")
    return httpx.Response(status_code=status_code, json=json_body or {}, request=request)


def _make_filter(
    *,
    excluded: list[str] | None = None,
    skip_encrypted: bool = True,
    skip_unlabeled: bool = False,
    http_client: MagicMock,
    logger: MagicMock,
) -> SensitivityLabelFilter:
    return SensitivityLabelFilter(
        excluded_label_ids=excluded or [],
        skip_encrypted=skip_encrypted,
        skip_unlabeled=skip_unlabeled,
        http_client=http_client,
        token_provider=AsyncMock(return_value="token"),
        logger=logger,
    )


# ---------------------------------------------------------------------------
# enabled flag
# ---------------------------------------------------------------------------


def test_disabled_when_empty(logger):
    """No block list and no skip_unlabeled means the filter is a no-op."""
    f = _make_filter(http_client=MagicMock(), logger=logger)
    assert f.enabled is False


def test_enabled_when_block_list_present(logger):
    f = _make_filter(excluded=["guid-1"], http_client=MagicMock(), logger=logger)
    assert f.enabled is True


def test_enabled_when_skip_unlabeled(logger):
    f = _make_filter(skip_unlabeled=True, http_client=MagicMock(), logger=logger)
    assert f.enabled is True


# ---------------------------------------------------------------------------
# Per-file (item) checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_item_with_blocked_label_is_skipped(logger):
    """An item whose extractSensitivityLabels response carries a blocked GUID is skipped."""
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(
            200,
            {"labels": [{"sensitivityLabelId": "GUID-1", "assignmentMethod": "standard"}]},
        )
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True


@pytest.mark.asyncio
async def test_item_label_match_is_case_insensitive(logger):
    """GUIDs from Graph and from config are compared case-insensitively."""
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(200, {"labels": [{"sensitivityLabelId": "AbCdEf"}]})
    )
    f = _make_filter(excluded=["abcdef"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True


@pytest.mark.asyncio
async def test_item_with_unrelated_label_is_kept(logger):
    """An item whose only labels are NOT in the block list is not skipped."""
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(200, {"labels": [{"sensitivityLabelId": "other-guid"}]})
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is False


@pytest.mark.asyncio
async def test_unlabeled_item_kept_by_default(logger):
    """labels:[] means unlabeled — kept when skip_unlabeled is False."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200, {"labels": []}))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is False


@pytest.mark.asyncio
async def test_unlabeled_item_skipped_when_skip_unlabeled(logger):
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200, {"labels": []}))
    f = _make_filter(excluded=[], skip_unlabeled=True, http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True


@pytest.mark.asyncio
async def test_encrypted_item_skipped_by_default(logger):
    """Graph 423 Locked → encrypted file → skipped when skip_encrypted is True (default)."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(423))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True


@pytest.mark.asyncio
async def test_encrypted_item_raises_when_skip_encrypted_false(logger):
    """With skip_encrypted=False the caller sees the 423 as an error."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(423))
    f = _make_filter(
        excluded=["guid-1"],
        skip_encrypted=False,
        http_client=http_client,
        logger=logger,
    )
    with pytest.raises(Exception, match="423"):
        await f.should_skip_item(drive_id="d", item_id="i")


@pytest.mark.asyncio
async def test_item_check_skipped_entirely_when_disabled(logger):
    """A disabled filter never calls Graph — no excluded labels, no skip flags."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200, {"labels": []}))
    f = _make_filter(http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is False
    http_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_graph_error_falls_back_to_keep(logger):
    """If extractSensitivityLabels errors (non-423), we keep the file rather than fail the sync."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(500))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is False


# ---------------------------------------------------------------------------
# Site (container) checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_site_with_blocked_container_label_is_skipped(logger):
    http_client = MagicMock()
    http_client.get = AsyncMock(
        return_value=_response(
            200, {"assignedLabels": [{"labelId": "GUID-1", "displayName": "Confidential"}]}
        )
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id="g") is True


@pytest.mark.asyncio
async def test_site_without_group_is_not_short_circuited(logger):
    """Non-group-connected sites have no group_id — fall through to per-file checks."""
    http_client = MagicMock()
    http_client.get = AsyncMock()
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id=None) is False
    http_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_site_with_unrelated_container_label_is_kept(logger):
    http_client = MagicMock()
    http_client.get = AsyncMock(
        return_value=_response(200, {"assignedLabels": [{"labelId": "other"}]})
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id="g") is False


@pytest.mark.asyncio
async def test_site_check_disabled_without_block_list(logger):
    """skip_unlabeled does not apply to sites — only the block list does."""
    http_client = MagicMock()
    http_client.get = AsyncMock()
    f = _make_filter(skip_unlabeled=True, http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id="g") is False
    http_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-label files and malformed Graph responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_item_with_one_blocked_label_among_many_is_skipped(logger):
    """Real files often carry several labels (parent + sublabel). One match is enough."""
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(
            200,
            {
                "labels": [
                    {"sensitivityLabelId": "harmless-1"},
                    {"sensitivityLabelId": "GUID-1"},
                    {"sensitivityLabelId": "harmless-2"},
                ]
            },
        )
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True


@pytest.mark.asyncio
async def test_item_with_all_unblocked_labels_is_kept(logger):
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(
            200,
            {
                "labels": [
                    {"sensitivityLabelId": "harmless-1"},
                    {"sensitivityLabelId": "harmless-2"},
                ]
            },
        )
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is False


@pytest.mark.asyncio
async def test_item_response_missing_labels_key_treated_as_unlabeled(logger):
    """A response without ``labels`` at all is treated as no labels — kept by default."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200, {}))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is False


@pytest.mark.asyncio
async def test_item_response_with_malformed_label_entries_ignored(logger):
    """Entries that aren't dicts or lack sensitivityLabelId must not crash the filter."""
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(
            200,
            {
                "labels": [
                    "not-a-dict",
                    {"assignmentMethod": "standard"},  # missing sensitivityLabelId
                    {"sensitivityLabelId": ""},  # empty
                    {"sensitivityLabelId": "GUID-1"},
                ]
            },
        )
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True


@pytest.mark.asyncio
async def test_site_with_empty_assigned_labels_is_kept(logger):
    """A group with assignedLabels:[] is unlabeled at the container level."""
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=_response(200, {"assignedLabels": []}))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id="g") is False


@pytest.mark.asyncio
async def test_site_response_missing_assigned_labels_key_is_kept(logger):
    """Group payload without assignedLabels (older tenants) — treated as unlabeled."""
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=_response(200, {}))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id="g") is False


@pytest.mark.asyncio
async def test_site_with_malformed_assigned_labels_ignored(logger):
    """Malformed entries in assignedLabels must not crash; valid ones still match."""
    http_client = MagicMock()
    http_client.get = AsyncMock(
        return_value=_response(
            200,
            {
                "assignedLabels": [
                    "not-a-dict",
                    {"displayName": "Confidential"},  # missing labelId
                    {"labelId": ""},
                    {"labelId": "GUID-1"},
                ]
            },
        )
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id="g") is True


@pytest.mark.asyncio
async def test_site_graph_error_falls_back_to_no_short_circuit(logger):
    """If reading group labels fails, the site is not pre-skipped — per-file still runs."""
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=_response(500))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_site(site_id="s", group_id="g") is False


# ---------------------------------------------------------------------------
# Configuration hygiene
# ---------------------------------------------------------------------------


def test_blank_and_whitespace_label_ids_filtered_at_init(logger):
    """Empty / whitespace-only GUIDs in config must be dropped, not used as wildcards."""
    f = _make_filter(
        excluded=["", "  ", "guid-1", "\t"],
        http_client=MagicMock(),
        logger=logger,
    )
    # Internal set should only contain the one real GUID, lowercased.
    assert f._excluded == {"guid-1"}


# ---------------------------------------------------------------------------
# Graph contract — URLs, params, headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_item_check_calls_correct_graph_url(logger):
    """Verify extractSensitivityLabels POSTs to the v1.0 driveItem endpoint."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200, {"labels": []}))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    await f.should_skip_item(drive_id="DRIVE-X", item_id="ITEM-Y")

    http_client.post.assert_called_once()
    called_url = http_client.post.call_args.args[0]
    assert called_url == (
        "https://graph.microsoft.com/v1.0/drives/DRIVE-X/items/ITEM-Y/extractSensitivityLabels"
    )
    # Authorization header must carry the bearer token from the provider.
    called_headers = http_client.post.call_args.kwargs["headers"]
    assert called_headers["Authorization"] == "Bearer token"
    assert called_headers["Accept"] == "application/json"


@pytest.mark.asyncio
async def test_site_check_calls_correct_graph_url_with_select(logger):
    """Verify assignedLabels are read off the M365 Group with $select=assignedLabels."""
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=_response(200, {"assignedLabels": []}))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    await f.should_skip_site(site_id="s", group_id="GROUP-Z")

    http_client.get.assert_called_once()
    called_url = http_client.get.call_args.args[0]
    assert called_url == "https://graph.microsoft.com/v1.0/groups/GROUP-Z"
    assert http_client.get.call_args.kwargs["params"] == {"$select": "assignedLabels"}
    called_headers = http_client.get.call_args.kwargs["headers"]
    assert called_headers["Authorization"] == "Bearer token"


@pytest.mark.asyncio
async def test_token_provider_called_per_graph_call(logger):
    """Token provider is invoked freshly per call — supports rotation/refresh."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(200, {"labels": []}))
    token_provider = AsyncMock(return_value="rotating-token")
    f = SensitivityLabelFilter(
        excluded_label_ids=["guid-1"],
        skip_encrypted=True,
        skip_unlabeled=False,
        http_client=http_client,
        token_provider=token_provider,
        logger=logger,
    )
    await f.should_skip_item(drive_id="d", item_id="i1")
    await f.should_skip_item(drive_id="d", item_id="i2")
    assert token_provider.call_count == 2
