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
from tenacity import wait_none

from airweave.platform.sources.microsoft_sensitivity_labels import SensitivityLabelFilter


@pytest.fixture
def logger():
    """Return a logger mock — the filter only needs .info/.warning/.error."""
    mock = MagicMock()
    mock.info = MagicMock()
    mock.warning = MagicMock()
    mock.error = MagicMock()
    return mock


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """Keep the filter's 429/5xx retries but skip their backoff sleeps."""
    for method in (SensitivityLabelFilter._graph_get, SensitivityLabelFilter._graph_post):
        monkeypatch.setattr(method.retry, "wait", wait_none())


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
# fails_closed flag — which policies refuse to index a file on an unread label
# ---------------------------------------------------------------------------


def test_skip_encrypted_alone_does_not_fail_closed(logger):
    """Encryption is reported as a 423; other read errors cannot hide plaintext."""
    f = _make_filter(http_client=MagicMock(), logger=logger)
    assert f.fails_closed is False


def test_block_list_fails_closed(logger):
    f = _make_filter(excluded=["guid-1"], http_client=MagicMock(), logger=logger)
    assert f.fails_closed is True


def test_skip_unlabeled_fails_closed(logger):
    f = _make_filter(skip_unlabeled=True, http_client=MagicMock(), logger=logger)
    assert f.fails_closed is True


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
async def test_default_config_reads_the_label_and_skips_an_encrypted_file(logger):
    """skip_encrypted_files defaults to true and is a policy on its own (review 4301738201).

    Regression: with no block list and skip_unlabeled off the filter was disabled, so an
    encrypted file never reached the 423 handling and was indexed.
    """
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(423))
    f = _make_filter(http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True
    http_client.post.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 429, 500, 503])
@pytest.mark.parametrize(
    "policy", [{"excluded": ["guid-1"]}, {"skip_unlabeled": True}], ids=["block_list", "unlabeled"]
)
async def test_unreadable_item_label_fails_closed_under_a_label_policy(logger, status, policy):
    """A label that cannot be read is not an allow decision (reviews 4301738207, 4311179790).

    Regression: any Graph error returned False (index the file) even with a block list, so
    a throttled sync indexed confidential-labelled files.
    """
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(status))
    f = _make_filter(http_client=http_client, logger=logger, **policy)
    with pytest.raises(httpx.HTTPStatusError):
        await f.should_skip_item(drive_id="d", item_id="i")
    logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_unreadable_item_label_is_kept_when_only_skip_encrypted_is_set(logger):
    """No policy decides on the label itself, so a non-423 read failure keeps the file."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(500))
    f = _make_filter(http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is False


@pytest.mark.asyncio
async def test_throttled_item_label_read_is_retried_before_deciding(logger):
    """A 429 is retried, so throttling alone does not fail a fail-closed sync."""
    http_client = MagicMock()
    http_client.post = AsyncMock(
        side_effect=[
            _response(429),
            _response(200, {"labels": [{"sensitivityLabelId": "GUID-1"}]}),
        ]
    )
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True
    assert http_client.post.call_count == 2


@pytest.mark.asyncio
async def test_encrypted_423_is_not_retried(logger):
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(423))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    assert await f.should_skip_item(drive_id="d", item_id="i") is True
    http_client.post.assert_called_once()


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
@pytest.mark.parametrize("status", [403, 429, 500])
async def test_site_graph_error_fails_closed_when_a_block_list_is_set(logger, status):
    """A site is not indexed on a container label that could not be checked (4301738204)."""
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=_response(status))
    f = _make_filter(excluded=["guid-1"], http_client=http_client, logger=logger)
    with pytest.raises(httpx.HTTPStatusError):
        await f.should_skip_site(site_id="s", group_id="g")


def test_only_a_block_list_blocks_containers(logger):
    f = _make_filter(skip_encrypted=False, http_client=MagicMock(), logger=logger)
    assert f.blocks_containers is False


@pytest.mark.parametrize(
    "module,cls",
    [
        ("airweave.platform.sources.onedrive", "OneDriveSource"),
        ("airweave.platform.sources.sharepoint", "SharePointSource"),
        ("airweave.platform.sources.sharepoint_online.source", "SharePointOnlineSource"),
    ],
)
@pytest.mark.parametrize(
    "excluded,skip_unlabeled,skip_encrypted",
    [
        ([], False, True),  # defaults: encrypted files must still be skipped (4301738201)
        ([], False, False),  # encrypted files must raise: the filter must exist
        (["g"], False, True),
        ([], True, True),
    ],
)
def test_the_source_builds_the_filter_for_every_label_setting(
    module, cls, excluded, skip_unlabeled, skip_encrypted, logger
):
    import importlib

    source = object.__new__(getattr(importlib.import_module(module), cls))
    source._label_filter = None
    source._excluded_sensitivity_label_ids = excluded
    source._skip_unlabeled_files = skip_unlabeled
    source._skip_encrypted_files = skip_encrypted
    source._http_client = MagicMock()
    source._logger = logger
    source.get_access_token = AsyncMock(return_value="t")

    assert source._get_label_filter() is not None


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


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded,raises", [(["guid-1"], True), ([], False)])
async def test_sharepoint_does_not_index_a_site_whose_label_check_failed(logger, excluded, raises):
    from airweave.platform.entities._base import Breadcrumb
    from airweave.platform.sources.sharepoint import SharePointSource

    source = object.__new__(SharePointSource)
    source._label_filter = None
    source._excluded_sensitivity_label_ids = excluded
    source._skip_unlabeled_files = False
    source._skip_encrypted_files = False  # a filter exists in both cases
    source._http_client = MagicMock()
    source._logger = logger
    source.get_access_token = AsyncMock(return_value="t")
    source._get = AsyncMock(side_effect=RuntimeError("graph down"))

    async def no_drives(*_a, **_k):
        return
        yield  # pragma: no cover

    source._generate_drive_entities = no_drives
    crumb = Breadcrumb(entity_id="s", name="S", entity_type="Site")

    async def run():
        return [e async for e in source._generate_drives_with_items("s", "S", crumb, 0)]

    if raises:
        with pytest.raises(RuntimeError):
            await run()
    else:
        assert await run() == []


# ---------------------------------------------------------------------------
# Source wiring — label decisions reach the entity stream
# ---------------------------------------------------------------------------

_FILE_ITEM = {
    "id": "item-1",
    "name": "plan.docx",
    "size": 10,
    "file": {"mimeType": "application/vnd.openxmlformats-officedocument"},
    "createdDateTime": "2026-01-01T00:00:00Z",
    "lastModifiedDateTime": "2026-01-02T00:00:00Z",
}


def _onedrive(logger, http_client, *, excluded=(), skip_unlabeled=False):
    from airweave.platform.sources.onedrive import OneDriveSource

    source = object.__new__(OneDriveSource)
    source._label_filter = None
    source._excluded_sensitivity_label_ids = list(excluded)
    source._skip_unlabeled_files = skip_unlabeled
    source._skip_encrypted_files = True  # the config default
    source._http_client = http_client
    source._logger = logger
    source.get_access_token = AsyncMock(return_value="t")

    async def one_file(_drive_id):
        yield dict(_FILE_ITEM)

    source._list_all_drive_items_recursively = one_file
    return source


async def _onedrive_entities(source) -> list:
    return [e async for e in source._generate_drive_item_entities("drive-1", "Drive")]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,kept", [(200, True), (423, False)])
async def test_onedrive_default_config_skips_an_encrypted_file(logger, status, kept):
    """Defaults (no block list, skip_encrypted_files=true) still drop a 423 file (4301738201)."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(status, {"labels": []}))
    entities = await _onedrive_entities(_onedrive(logger, http_client))
    assert len(entities) == (1 if kept else 0)
    http_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_onedrive_does_not_index_a_file_whose_label_read_failed(logger):
    """With a block list, an unreadable label fails the sync instead of indexing (4301738207)."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(403))
    source = _onedrive(logger, http_client, excluded=["guid-1"])
    with pytest.raises(httpx.HTTPStatusError):
        await _onedrive_entities(source)


def _sharepoint_online(logger, http_client, *, excluded=()):
    from airweave.platform.configs.config import SharePointOnlineConfig
    from airweave.platform.sources.sharepoint_online.source import SharePointOnlineSource

    source = object.__new__(SharePointOnlineSource)
    source._init_common(SharePointOnlineConfig(excluded_sensitivity_label_ids=list(excluded)))
    source._http_client = http_client
    source._logger = logger
    source._get_access_token = AsyncMock(return_value="t")
    return source


@pytest.mark.asyncio
async def test_sharepoint_online_does_not_index_a_file_whose_label_read_failed(logger):
    """The SPO drive walk propagates an unreadable label under a block list (4301738207)."""
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_response(503))
    source = _sharepoint_online(logger, http_client, excluded=["guid-1"])
    graph_client = MagicMock()
    graph_client.get_item_permissions = AsyncMock(return_value=[])

    async def items():
        yield dict(_FILE_ITEM)

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in source._process_file_items(
            graph_client, items(), "drive-1", "site-1", [], None, site_url="https://t/sites/a"
        ):
            pass
    graph_client.get_item_permissions.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 429, 500])
async def test_sharepoint_online_does_not_index_a_site_whose_container_label_failed(logger, status):
    """A failed container-label read under a block list stops the site (4301738204)."""
    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=_response(status))
    source = _sharepoint_online(logger, http_client, excluded=["guid-1"])
    site = {"id": "site-1", "displayName": "HR", "webUrl": "https://t.sharepoint.com/sites/hr"}
    source._discover_sites = AsyncMock(return_value=[site])
    graph_client = MagicMock()

    async def drives(_site_id):
        yield {"id": "drive-1", "name": "Docs", "owner": {"group": {"id": "group-1"}}}

    graph_client.get_drives = drives
    graph_client.get_drive_root_permissions = AsyncMock(return_value=[])
    source._create_graph_client = MagicMock(return_value=graph_client)

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in source._full_sync(None, None):
            pass
    graph_client.get_drive_root_permissions.assert_not_called()
