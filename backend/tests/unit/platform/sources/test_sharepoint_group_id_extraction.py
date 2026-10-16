"""Tests for the M365 Group ID extraction used by the sensitivity-label site short-circuit.

Both the simple ``SharePointSource`` and the feature-rich
``SharePointOnlineBase`` source expose a static
``_extract_group_id_from_drives`` that digs the backing Microsoft 365 Group
GUID out of any drive's ``owner.group.id``. The result feeds
``SensitivityLabelFilter.should_skip_site`` for container-label checks.

If extraction is wrong, the site short-circuit either never fires
(per-file checks still cover us, but we do an extra Graph call per file)
or fires against the wrong group (we'd skip the wrong site). Both worth
locking down.
"""

from __future__ import annotations

import pytest

from airweave.platform.sources.sharepoint import SharePointSource
from airweave.platform.sources.sharepoint_online.source import SharePointOnlineBase

# The two sources have identical helpers — parametrize so every assertion
# runs against both implementations.
EXTRACTORS = [
    pytest.param(SharePointSource._extract_group_id_from_drives, id="sharepoint"),
    pytest.param(SharePointOnlineBase._extract_group_id_from_drives, id="sharepoint_online"),
]


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_returns_group_id_from_first_drive(extract):
    drives = [
        {"id": "d1", "owner": {"group": {"id": "group-guid-1"}}},
    ]
    assert extract(drives) == "group-guid-1"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_walks_drives_until_one_has_a_group_owner(extract):
    """Walk past user-owned drives until we find one owned by the M365 Group.

    Some drives in a site are personal libraries; those don't carry the group.
    """
    drives = [
        {"id": "d1", "owner": {"user": {"id": "u-1"}}},
        {"id": "d2", "owner": {"group": {"id": "group-guid-1"}}},
    ]
    assert extract(drives) == "group-guid-1"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_returns_none_when_no_drive_has_a_group_owner(extract):
    """Classic comms sites and personal OneDrive sites lack a backing group."""
    drives = [
        {"id": "d1", "owner": {"user": {"id": "u-1"}}},
        {"id": "d2", "owner": {"user": {"id": "u-2"}}},
    ]
    assert extract(drives) is None


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_returns_none_for_empty_drives_list(extract):
    assert extract([]) is None


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_handles_drives_missing_owner_field(extract):
    """Some Graph responses omit ``owner`` entirely; the helper must not crash."""
    drives = [
        {"id": "d1"},
        {"id": "d2", "owner": None},
        {"id": "d3", "owner": {"group": {"id": "group-guid-1"}}},
    ]
    assert extract(drives) == "group-guid-1"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_handles_owner_with_no_group_subobject(extract):
    drives = [
        {"id": "d1", "owner": {}},
        {"id": "d2", "owner": {"group": None}},
        {"id": "d3", "owner": {"group": {"id": "group-guid-1"}}},
    ]
    assert extract(drives) == "group-guid-1"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_handles_group_missing_id(extract):
    """A ``group`` block without an ``id`` is malformed — fall through."""
    drives = [
        {"id": "d1", "owner": {"group": {"displayName": "no-id"}}},
        {"id": "d2", "owner": {"group": {"id": "group-guid-1"}}},
    ]
    assert extract(drives) == "group-guid-1"
