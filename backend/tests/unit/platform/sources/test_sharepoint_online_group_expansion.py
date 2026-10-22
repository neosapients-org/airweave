"""Unit tests for SharePoint Online SP site group expansion helpers.

Covers _parse_sp_group_member, _email_from_membership_login, the cursor
migration path for tracked_sp_groups, site-scoped SP group keys, group tracking
on the incremental path, and the tenant-user enumeration's failure handling.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

from airweave.domains.syncs.cursors.cursor import SyncCursor
from airweave.platform.cursors.sharepoint_online import SharePointOnlineCursor
from airweave.platform.sources.sharepoint_online.acl import (
    extract_access_control,
    format_sp_group_id,
)
from airweave.platform.sources.sharepoint_online.source import (
    EVERYONE_EXCEPT_EXTERNAL_PRINCIPAL,
    SharePointOnlineBase,
)

# ---------------------------------------------------------------------------
# _email_from_membership_login
# ---------------------------------------------------------------------------


def test_email_from_membership_login_valid():
    assert (
        SharePointOnlineBase._email_from_membership_login("i:0#.f|membership|foo@bar.com")
        == "foo@bar.com"
    )


def test_email_from_membership_login_uppercase_normalized():
    assert (
        SharePointOnlineBase._email_from_membership_login("i:0#.f|membership|Foo@BAR.com")
        == "foo@bar.com"
    )


def test_email_from_membership_login_rejects_role_principal():
    # Role principals would otherwise yield "spo-grid-all-users/..." — must reject.
    assert (
        SharePointOnlineBase._email_from_membership_login(
            "c:0-.f|rolemanager|spo-grid-all-users/26adf163-2699-4d04-a0ad-3d935411bf45"
        )
        is None
    )


def test_email_from_membership_login_rejects_federated_group():
    assert (
        SharePointOnlineBase._email_from_membership_login(
            "c:0o.c|federateddirectoryclaimprovider|58cb1814-203a-44d0-8578-b53f63860579"
        )
        is None
    )


def test_email_from_membership_login_rejects_empty():
    assert SharePointOnlineBase._email_from_membership_login("") is None


def test_email_from_membership_login_rejects_malformed():
    assert SharePointOnlineBase._email_from_membership_login("i:0#.f|membership|noat") is None


# ---------------------------------------------------------------------------
# _parse_sp_group_member
# ---------------------------------------------------------------------------


def test_parse_real_user_with_email():
    user = {
        "PrincipalType": 1,
        "LoginName": "i:0#.f|membership|alice@contoso.com",
        "Email": "alice@contoso.com",
        "Title": "Alice",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        "alice@contoso.com",
        "user",
    )


def test_parse_real_user_uppercase_email_normalized():
    user = {
        "PrincipalType": 1,
        "LoginName": "i:0#.f|membership|ALICE@CONTOSO.COM",
        "Email": "ALICE@CONTOSO.COM",
        "Title": "Alice",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        "alice@contoso.com",
        "user",
    )


def test_parse_real_user_email_empty_fallback_to_login():
    # If Email is missing but LoginName has the membership pattern, use that.
    user = {
        "PrincipalType": 1,
        "LoginName": "i:0#.f|membership|alice@contoso.com",
        "Email": "",
        "Title": "Alice",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        "alice@contoso.com",
        "user",
    )


def test_parse_real_user_no_email_no_parseable_login_returns_none():
    # System Account and similar — no Email, no membership LoginName.
    user = {
        "PrincipalType": 1,
        "LoginName": "SHAREPOINT\\system",
        "Email": "",
        "Title": "System Account",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_role_principal_skipped():
    """Bug B regression test — 'Everyone except external users' must not become a fake user."""
    user = {
        "PrincipalType": 16,
        "LoginName": "c:0-.f|rolemanager|spo-grid-all-users/26adf163-2699-4d04-a0ad-3d935411bf45",
        "Email": "",
        "Title": "Everyone except external users",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_entra_group_emits_group_membership():
    """Bug C/D regression test — Entra group must be emitted as group-to-group."""
    user = {
        "PrincipalType": 4,
        "LoginName": "c:0o.c|federateddirectoryclaimprovider|58cb1814-203a-44d0-8578-b53f63860579",
        "Email": "neena@neenacorp.onmicrosoft.com",  # group's email, must NOT be used
        "Title": "Neena Members",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        "entra:58cb1814-203a-44d0-8578-b53f63860579",
        "group",
    )


def test_parse_entra_group_owner_suffix_stripped():
    """Owner-style claim has `_o` suffix — must strip it to get the bare GUID."""
    login = "c:0o.c|federateddirectoryclaimprovider|58cb1814-203a-44d0-8578-b53f63860579_o"
    user = {
        "PrincipalType": 4,
        "LoginName": login,
        "Email": "neena@neenacorp.onmicrosoft.com",
        "Title": "Neena Owners",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        "entra:58cb1814-203a-44d0-8578-b53f63860579",
        "group",
    )


def test_parse_entra_group_uppercase_guid_normalized():
    user = {
        "PrincipalType": 4,
        "LoginName": "c:0o.c|federateddirectoryclaimprovider|58CB1814-203A-44D0-8578-B53F63860579",
        "Title": "Neena Owners",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        "entra:58cb1814-203a-44d0-8578-b53f63860579",
        "group",
    )


def test_parse_entra_group_malformed_guid_returns_none():
    user = {
        "PrincipalType": 4,
        "LoginName": "c:0o.c|federateddirectoryclaimprovider|not-a-guid",
        "Title": "Bad Group",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_security_group_non_federated_returns_none():
    # PrincipalType=4 but not federated — on-prem AD claim, skip.
    user = {
        "PrincipalType": 4,
        "LoginName": "c:0-.f|adclaimprovider|S-1-5-21-...",
        "Title": "On-prem Group",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_everyone_except_external_claim_returns_synthetic_sentinel():
    """The rolemanager/spo-grid-all-users claim → synthetic group sentinel."""
    user = {
        "PrincipalType": 4,
        "LoginName": ("c:0-.f|rolemanager|spo-grid-all-users/26adf163-2699-4d04-a0ad-3d935411bf45"),
        "Title": "Everyone except external users",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        EVERYONE_EXCEPT_EXTERNAL_PRINCIPAL,
        "group",
    )


def test_parse_everyone_except_external_uppercase_tenant_id():
    """Tenant ID GUIDs may be upper- or lowercase; both should match."""
    user = {
        "PrincipalType": 4,
        "LoginName": ("c:0-.f|rolemanager|spo-grid-all-users/26ADF163-2699-4D04-A0AD-3D935411BF45"),
        "Title": "Everyone except external users",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) == (
        EVERYONE_EXCEPT_EXTERNAL_PRINCIPAL,
        "group",
    )


def test_parse_other_rolemanager_claim_skipped_and_flagged_as_unrecognized():
    """A different rolemanager claim shouldn't match; should be flagged for logging."""
    user = {
        "PrincipalType": 4,
        "LoginName": "c:0-.f|rolemanager|some-future-claim",
        "Title": "Custom Role",
    }
    assert SharePointOnlineBase._parse_sp_group_member(user) is None
    assert SharePointOnlineBase._is_unrecognized_pt4_login(user) is True


def test_is_unrecognized_pt4_login_false_for_known_shapes():
    """Known PT=4 shapes (Entra group, claim) must NOT be flagged as unrecognized."""
    entra = {
        "PrincipalType": 4,
        "LoginName": (
            "c:0o.c|federateddirectoryclaimprovider|7d344400-39bc-4ee7-aa6e-437bd8de85c0"
        ),
    }
    claim = {
        "PrincipalType": 4,
        "LoginName": "c:0-.f|rolemanager|spo-grid-all-users/26adf163-2699-4d04-a0ad-3d935411bf45",
    }
    assert SharePointOnlineBase._is_unrecognized_pt4_login(entra) is False
    assert SharePointOnlineBase._is_unrecognized_pt4_login(claim) is False


def test_is_unrecognized_pt4_login_false_for_non_pt4():
    """The flag is scoped to PT=4 only; other PrincipalTypes are skipped silently."""
    user = {"PrincipalType": 1, "LoginName": "i:0#.f|membership|alice@example.com"}
    assert SharePointOnlineBase._is_unrecognized_pt4_login(user) is False


def test_parse_distlist_skipped():
    user = {"PrincipalType": 2, "LoginName": "some-dl", "Title": "DL"}
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_spgroup_skipped():
    user = {"PrincipalType": 8, "LoginName": "some-sp", "Title": "SP"}
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_all_catchall_skipped():
    user = {"PrincipalType": 15, "LoginName": "everyone", "Title": "All"}
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_unknown_principal_type_skipped():
    user = {"PrincipalType": 99, "LoginName": "x", "Title": "X"}
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


def test_parse_missing_principal_type_skipped():
    user = {"LoginName": "x", "Title": "X", "Email": "x@y.z"}
    assert SharePointOnlineBase._parse_sp_group_member(user) is None


# ---------------------------------------------------------------------------
# _normalize_site_url
# ---------------------------------------------------------------------------


def test_normalize_site_url_strips_trailing_slash():
    assert (
        SharePointOnlineBase._normalize_site_url("https://contoso.sharepoint.com/sites/X/")
        == "https://contoso.sharepoint.com/sites/X"
    )


def test_normalize_site_url_empty():
    assert SharePointOnlineBase._normalize_site_url("") == ""
    assert SharePointOnlineBase._normalize_site_url(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _track_entity_groups with site_url scoping
# ---------------------------------------------------------------------------


class _StubEntity:
    def __init__(self, viewers):
        self.access = MagicMock()
        self.access.viewers = viewers


def _bare_base() -> SharePointOnlineBase:
    """Instantiate the base class just enough to exercise tracking logic.

    We bypass the normal source creation path since we only need the tracking
    state and its methods.
    """
    instance = SharePointOnlineBase.__new__(SharePointOnlineBase)
    instance._site_url = ""
    instance._include_personal_sites = False
    instance._include_pages = False
    instance._item_level_entra_groups = set()
    instance._item_level_sp_groups = {}
    instance._needs_internal_user_enum = False
    instance._drive_site_urls = {}
    instance._logger = MagicMock()
    return instance


def test_track_entity_groups_scopes_sp_by_site():
    base = _bare_base()
    e = _StubEntity(
        [
            "group:sp:neena_members",
            "group:sp:neena_owners",
            "group:entra:58cb1814-203a-44d0-8578-b53f63860579",
            "user:alice@contoso.com",
        ]
    )
    base._track_entity_groups(e, "https://neenacorp.sharepoint.com/sites/Neena77")

    assert base._item_level_sp_groups == {
        "https://neenacorp.sharepoint.com/sites/Neena77": {
            "sp:neena_members",
            "sp:neena_owners",
        }
    }
    assert base._item_level_entra_groups == {"entra:58cb1814-203a-44d0-8578-b53f63860579"}


def test_track_entity_groups_multiple_sites_keep_separate():
    base = _bare_base()
    base._track_entity_groups(
        _StubEntity(["group:sp:neena_members"]),
        "https://neenacorp.sharepoint.com/sites/A",
    )
    base._track_entity_groups(
        _StubEntity(["group:sp:access_control_tests_owners"]),
        "https://neenacorp.sharepoint.com/sites/B",
    )

    assert base._item_level_sp_groups == {
        "https://neenacorp.sharepoint.com/sites/A": {"sp:neena_members"},
        "https://neenacorp.sharepoint.com/sites/B": {"sp:access_control_tests_owners"},
    }


def test_track_entity_groups_same_name_different_sites_do_not_collide():
    base = _bare_base()
    base._track_entity_groups(
        _StubEntity(["group:sp:members"]),
        "https://neenacorp.sharepoint.com/sites/A",
    )
    base._track_entity_groups(
        _StubEntity(["group:sp:members"]),
        "https://neenacorp.sharepoint.com/sites/B",
    )

    # Same group name but two different sites — must be tracked independently.
    assert set(base._item_level_sp_groups.keys()) == {
        "https://neenacorp.sharepoint.com/sites/A",
        "https://neenacorp.sharepoint.com/sites/B",
    }


def test_track_entity_groups_normalizes_trailing_slash():
    base = _bare_base()
    base._track_entity_groups(
        _StubEntity(["group:sp:x"]),
        "https://neenacorp.sharepoint.com/sites/A/",
    )
    base._track_entity_groups(
        _StubEntity(["group:sp:y"]),
        "https://neenacorp.sharepoint.com/sites/A",
    )
    # Both should land under the same normalized key.
    assert base._item_level_sp_groups == {
        "https://neenacorp.sharepoint.com/sites/A": {"sp:x", "sp:y"}
    }


def test_track_entity_groups_no_access_noop():
    base = _bare_base()
    entity = MagicMock()
    entity.access = None
    base._track_entity_groups(entity, "https://neenacorp.sharepoint.com/sites/A")
    assert base._item_level_sp_groups == {}


def test_track_entity_groups_without_a_site_does_not_track_sp_groups():
    """An SP group without its site cannot be expanded, so it is not tracked or persisted."""
    base = _bare_base()
    base._track_entity_groups(_StubEntity(["group:sp:orphan", "group:entra:g1"]), "")
    assert base._item_level_sp_groups == {}
    assert base._item_level_entra_groups == {"entra:g1"}


# ---------------------------------------------------------------------------
# Site-scoped SP group keys end to end (review 4301738218)
# ---------------------------------------------------------------------------

SITE_A = "https://t.sharepoint.com/sites/A"
SITE_B = "https://t.sharepoint.com/sites/B"


def _members_perm() -> dict:
    return {
        "id": "sg-3",
        "roles": ["read"],
        "grantedToV2": {"siteGroup": {"displayName": "Members", "id": "3"}},
    }


def _sp_user(email: str) -> dict:
    return {"PrincipalType": 1, "LoginName": f"i:0#.f|membership|{email}", "Email": email}


@pytest.mark.asyncio
async def test_same_named_groups_on_two_sites_keep_their_own_members():
    """Viewer and membership use one site-scoped key; two "Members" groups never merge.

    Regression: both sides were ``sp:members``, so one site's members could read the
    other site's files.
    """
    base = _bare_base()
    graph = MagicMock()
    graph.get_site_groups = AsyncMock(return_value=[{"Title": "Members", "Id": 3}])
    users = {SITE_A: [_sp_user("alice@t.com")], SITE_B: [_sp_user("bob@t.com")]}
    graph.get_site_group_users = AsyncMock(side_effect=lambda url, _id, **_kw: users[url])
    base._create_graph_client = MagicMock(return_value=graph)
    base._make_sp_token_provider_for_site = MagicMock(return_value=AsyncMock())

    viewers = {}
    for site in (SITE_A, SITE_B):
        access = await extract_access_control([_members_perm()], site_url=site)
        viewers[site] = access.viewers[0]
        entity = _StubEntity(access.viewers)
        base._track_entity_groups(entity, site)

    memberships = {(m.member_id, m.group_id) async for m in base._expand_sp_site_groups()}

    assert viewers[SITE_A] != viewers[SITE_B]
    assert memberships == {
        ("alice@t.com", viewers[SITE_A][len("group:") :]),
        ("bob@t.com", viewers[SITE_B][len("group:") :]),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tracked",
    [
        {SITE_A: ["sp:members"]},  # pre-scoping key
        {"": ["sp:members"]},  # tracked without a site
        ["sp:members"],  # oldest list format
    ],
)
async def test_a_cursor_with_unscoped_sp_group_keys_forces_a_full_sync(tracked):
    """Entities indexed under unscoped viewers are re-indexed before memberships switch."""
    base = _bare_base()
    ran = []

    async def full(_cursor, _files):
        ran.append("full")
        return
        yield  # pragma: no cover

    async def incremental(_cursor, _files):
        ran.append("incremental")
        return
        yield  # pragma: no cover

    base._full_sync = full
    base._incremental_sync = incremental
    base._should_do_full_sync = MagicMock(return_value=(False, "incremental"))
    cursor = MagicMock()
    cursor.data = {"tracked_sp_groups": tracked}

    async for _ in base.generate_entities(cursor=cursor):
        pass

    assert ran == ["full"]
    assert base._item_level_sp_groups == {}


@pytest.mark.asyncio
async def test_a_cursor_with_scoped_sp_group_keys_stays_incremental():
    base = _bare_base()
    ran = []

    async def incremental(_cursor, _files):
        ran.append("incremental")
        return
        yield  # pragma: no cover

    base._incremental_sync = incremental
    base._should_do_full_sync = MagicMock(return_value=(False, "incremental"))
    cursor = MagicMock()
    cursor.data = {"tracked_sp_groups": {SITE_A: [format_sp_group_id(SITE_A, "Members")]}}

    async for _ in base.generate_entities(cursor=cursor):
        pass

    assert ran == ["incremental"]


# ---------------------------------------------------------------------------
# Incremental path tracks SP groups under the drive's site (review 4301738222)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_sync_tracks_sp_groups_under_the_drive_site_and_persists_them():
    """A group first seen on a delta run is expandable now and on the next run.

    Regression: the delta path tracked SP groups under the empty site key, which
    expansion skips, and never wrote them to the cursor.
    """
    base = _bare_base()
    graph = MagicMock()
    item = {"id": "item-1", "name": "a.docx", "file": {"mimeType": "x"}, "size": 1}
    graph.get_drive_delta = AsyncMock(return_value=([item], "https://delta/next"))
    graph.get_item_permissions = AsyncMock(return_value=[_members_perm()])
    graph.get_drive_site_url = AsyncMock(return_value="https://t.sharepoint.com/sites/HR/")
    base._create_graph_client = MagicMock(return_value=graph)
    base._label_filter = MagicMock()
    base._label_filter.should_skip_item = AsyncMock(return_value=False)
    cursor = SyncCursor(
        sync_id=uuid4(),
        cursor_schema=SharePointOnlineCursor,
        cursor_data={"drive_delta_tokens": {"drive-1": "https://delta/prev"}},
    )

    entities = [e async for e in base._incremental_sync(cursor, None)]

    site = "https://t.sharepoint.com/sites/HR"
    group_id = format_sp_group_id(site, "Members")
    assert entities[0].access.viewers == [f"group:{group_id}"]
    assert base._item_level_sp_groups == {site: {group_id}}
    assert cursor.data["tracked_sp_groups"] == {site: [group_id]}


# ---------------------------------------------------------------------------
# Tenant-user enumeration must not complete truncated (review 4301738224)
# ---------------------------------------------------------------------------


def _failing_user_pages(graph: MagicMock) -> None:
    async def users():
        yield {"email": "a@t.com", "display_name": "A"}
        yield {"email": "b@t.com", "display_name": "B"}
        request = httpx.Request("GET", "https://graph.microsoft.com/v1.0/users")
        raise httpx.HTTPStatusError(
            "boom", request=request, response=httpx.Response(503, request=request)
        )

    graph.list_internal_tenant_users = users


@pytest.mark.asyncio
async def test_tenant_user_enumeration_reraises_after_a_partial_page():
    """A later-page failure re-raises after the users already yielded.

    Regression: the error was logged and the stream completed, so the membership sync
    saw a complete-looking, truncated set and deleted the rest as orphans.
    """
    base = _bare_base()
    graph = MagicMock()
    _failing_user_pages(graph)
    base._create_graph_client = MagicMock(return_value=graph)

    seen = []
    with pytest.raises(httpx.HTTPStatusError):
        async for m in base._expand_everyone_except_external():
            seen.append(m.member_id)
    assert seen == ["a@t.com", "b@t.com"]


@pytest.mark.asyncio
async def test_membership_stream_fails_when_tenant_user_enumeration_fails():
    """The failure reaches the ACL pipeline, which then skips orphan cleanup."""
    base = _bare_base()
    graph = MagicMock()
    _failing_user_pages(graph)
    base._create_graph_client = MagicMock(return_value=graph)
    base._create_group_expander = MagicMock(return_value=MagicMock())

    async def no_entra(_expander):
        return
        yield  # pragma: no cover

    async def sp_groups_using_the_claim():
        base._needs_internal_user_enum = True
        return
        yield  # pragma: no cover

    base._expand_entra_groups = no_entra
    base._expand_sp_site_groups = sp_groups_using_the_claim

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in base.generate_access_control_memberships():
            pass
