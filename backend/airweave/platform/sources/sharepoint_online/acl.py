"""Access control helpers for SharePoint Online.

Maps Microsoft Graph API permissions to Airweave's AccessControl format.

Graph permission model:
- grantedToV2.user → user:{email}
- grantedToV2.group (Entra ID) → group:entra:{group_id}
- grantedToV2.siteGroup → group:sp:{site_scope}:{site_group_name}
- link with scope "anonymous" → is_public (true tenant-wide / internet-wide access)
- link with scope "organization" or "users" →
  group:sp:{site_scope}:sharinglinks.{itemId}.{scopeRole}.{linkId}
  derived from the permission and the file's SP UniqueId. Microsoft represents
  these links as a ``link`` permission rather than a ``siteGroup`` grant, but
  internally tracks redeemers as members of a ``SharingLinks.<itemId>.<scopeRole>.<linkId>``
  SP site group. We translate so that membership intersection at search time
  works for users who have actually redeemed the link. The explicit recipients
  Graph lists on the link (``grantedToIdentitiesV2``) are kept as viewers too.

SharePoint site groups are scoped to their site: two sites can each have a group
called "Members" with different members. ``{site_scope}`` (see :func:`sp_site_scope`)
keeps their keys apart; the entity viewer and the membership ``group_id`` are both
built by :func:`format_sp_group_id`, so the two sides always agree.
"""

import hashlib
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from airweave.platform.entities._base import AccessControl


def sp_site_scope(site_url: str) -> str:
    """Return the deterministic site component of an SP group key.

    SharePoint URLs are case-insensitive, so the host and the decoded path are lowercased
    and the scheme and trailing slash dropped before hashing. A 16-hex digest keeps the
    key short enough for the 255-character membership ``group_id`` column.

    Args:
        site_url: The site's absolute URL (``https://tenant.sharepoint.com/sites/X``).

    Returns:
        The site scope, or ``""`` if the URL has no host.
    """
    parsed = urlparse((site_url or "").strip())
    host = parsed.netloc.lower()
    if not host:
        return ""
    canonical = f"{host}{unquote(parsed.path).rstrip('/').lower()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def format_sp_group_id(site_url: str, group_name: str) -> Optional[str]:
    """Build the membership ``group_id`` of a SharePoint site group.

    Membership group_id: ``sp:{site_scope}:{name}``; entity viewer: ``group:`` + that.

    Args:
        site_url: URL of the site the group belongs to.
        group_name: The group's title (spaces become underscores, lowercased).

    Returns:
        The group id, or None if the site or the name is unknown: a group that cannot be
        tied to its site is never matched, so it grants nothing.
    """
    scope = sp_site_scope(site_url)
    if not scope or not group_name:
        return None
    return f"sp:{scope}:{group_name.lower().replace(' ', '_')}"


def _resolve_user_principal(user: Dict[str, Any]) -> Optional[str]:
    """Resolve a Graph user identity to a canonical principal string."""
    for field in ("email", "userPrincipalName", "displayName"):
        val = user.get(field, "")
        if val and "@" in val:
            return f"user:{val.lower()}"
    user_id = user.get("id", "")
    return f"user:id:{user_id}" if user_id else None


def _resolve_group_principal(group: Dict[str, Any]) -> Optional[str]:
    """Resolve a Graph group identity to a canonical principal string."""
    group_id = group.get("id", "")
    return f"group:entra:{group_id}" if group_id else None


def _resolve_site_group_principal(site_group: Dict[str, Any], site_url: str) -> Optional[str]:
    """Resolve a SP site group identity to a site-scoped principal string."""
    sp_id = site_group.get("id")
    name = site_group.get("displayName", "") or (str(sp_id) if sp_id else "")
    group_id = format_sp_group_id(site_url, name)
    return f"group:{group_id}" if group_id else None


def extract_principal_from_permission(permission: Dict[str, Any], site_url: str) -> Optional[str]:
    """Extract a canonical principal ID from a Graph permission object.

    Args:
        permission: Graph API permission dict with grantedToV2, roles, link, etc.
        site_url: URL of the site the permission belongs to (scopes SP site groups).

    Returns:
        Canonical principal string or None if not resolvable.
    """
    granted_to = permission.get("grantedToV2") or permission.get("grantedTo")
    if not granted_to:
        return None

    user = granted_to.get("user")
    if user:
        return _resolve_user_principal(user)

    group = granted_to.get("group")
    if group:
        return _resolve_group_principal(group)

    site_group = granted_to.get("siteGroup")
    if site_group:
        return _resolve_site_group_principal(site_group, site_url)

    return None


def has_read_permission(permission: Dict[str, Any]) -> bool:
    """Check if a permission grants at least read access."""
    roles = permission.get("roles", [])
    return any(r in ("read", "write", "owner", "sp.full control") for r in roles)


def is_anonymous_link(permission: Dict[str, Any]) -> bool:
    """Check if a permission is an anonymous sharing link."""
    link = permission.get("link")
    if not link:
        return False
    return link.get("scope", "") == "anonymous"


# Mapping from Graph (link.scope, link.type) to SharePoint's SharingLinks group
# suffix. Verified empirically against neenacorp.sharepoint.com:
#   organization+edit → OrganizationEdit
#   organization+view → OrganizationView
#   users+edit / users+view → Flexible  (both collapse; SP stores role separately)
# Anonymous is handled by ``is_public`` and does not need a derived group.
_SCOPE_ROLE_MAP: Dict[tuple, str] = {
    ("organization", "edit"): "OrganizationEdit",
    ("organization", "view"): "OrganizationView",
    ("users", "edit"): "Flexible",
    ("users", "view"): "Flexible",
}


def link_permission_to_sp_group_viewer(
    permission: Dict[str, Any], sp_unique_id: Optional[str], site_url: str
) -> Optional[str]:
    """Derive the SharingLinks SP site group viewer for a non-anonymous link permission.

    SharePoint creates an internal site group named
    ``SharingLinks.<fileSpUniqueId>.<ScopeRole>.<linkId>`` for each sharing
    link, whose members are the users who have redeemed the link. The Graph
    per-item permissions response represents the link itself but does *not*
    return that site group as a separate ``siteGroup`` grant, so we translate.

    Args:
        permission: A Graph permission with a ``link`` block.
        sp_unique_id: The file's SharePoint UniqueId (lowercase GUID, no
            braces). Pass ``None`` for site/drive-level permissions, where
            sharing-link translation does not apply.
        site_url: URL of the site the file lives in (scopes the SharingLinks group).

    Returns:
        ``group:sp:<site_scope>:sharinglinks.<id>.<scoperole>.<linkid>`` viewer
        string, or ``None`` if the permission isn't a translatable link or
        required fields are missing.
    """
    if not sp_unique_id:
        return None
    link = permission.get("link")
    if not link:
        return None
    scope = link.get("scope", "")
    if scope == "anonymous":
        return None  # handled by is_public
    scope_role = _SCOPE_ROLE_MAP.get((scope, link.get("type", "")))
    if not scope_role:
        return None  # unknown scope/type combination — be conservative
    link_id = permission.get("id", "")
    if not link_id:
        return None
    group_id = format_sp_group_id(site_url, f"SharingLinks.{sp_unique_id}.{scope_role}.{link_id}")
    return f"group:{group_id}" if group_id else None


def _extract_identity_principals(perm: Dict[str, Any], viewers: List[str]) -> None:
    """Extract user principals from grantedToIdentitiesV2/grantedToIdentities."""
    for identities_key in ("grantedToIdentitiesV2", "grantedToIdentities"):
        for identity in perm.get(identities_key, []):
            user = identity.get("user")
            if not user:
                continue
            pid = _resolve_user_principal(user)
            if pid and pid not in viewers:
                viewers.append(pid)


async def extract_access_control(
    permissions: List[Dict[str, Any]],
    sp_unique_id: Optional[str] = None,
    *,
    site_url: str,
) -> AccessControl:
    """Build AccessControl from Graph API permissions.

    Args:
        permissions: List of permission objects from Graph API.
        sp_unique_id: The SharePoint UniqueId of the item the permissions
            belong to (lowercase GUID, no braces). Required to translate
            non-anonymous sharing-link permissions into their corresponding
            ``SharingLinks.*`` SP site group viewer. Pass ``None`` for
            site/drive-level permission lists.
        site_url: URL of the site the permissions belong to. Every SP site group
            viewer is scoped to it; with an empty URL no SP group grants access.

    Returns:
        AccessControl with viewers and is_public flag.
    """
    viewers: List[str] = []
    is_public = False

    for perm in permissions:
        if not has_read_permission(perm):
            continue

        if is_anonymous_link(perm):
            is_public = True
            continue

        # Non-anonymous sharing links: translate to the per-link SP site group, and keep
        # the recipients Graph names on the link — a ``users`` link's invitee who has not
        # redeemed it yet is not a member of the SharingLinks group.
        link_viewer = link_permission_to_sp_group_viewer(perm, sp_unique_id, site_url)
        if link_viewer:
            if link_viewer not in viewers:
                viewers.append(link_viewer)
            _extract_identity_principals(perm, viewers)
            continue

        principal = extract_principal_from_permission(perm, site_url)
        if principal and principal not in viewers:
            viewers.append(principal)

        _extract_identity_principals(perm, viewers)

    return AccessControl(viewers=viewers, is_public=is_public)


def format_entra_group_id(group_id: str) -> str:
    """Format Entra ID group ID for membership records.

    Membership group_id: "entra:{group_id}"
    Entity viewer: "group:entra:{group_id}"
    """
    return f"entra:{group_id}"
