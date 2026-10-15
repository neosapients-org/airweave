"""Centralized path constants for Airweave storage operations.

All temp and persistent storage paths should be defined here for consistency.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

_UNSAFE_CHARS_RE = re.compile(r'[/\\:*?"<>|]')
_MULTI_UNDERSCORE_RE = re.compile(r"_+")
_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:[\\/]")


class StoragePaths:
    """Centralized storage path constants and builders."""

    # =========================================================================
    # Base directories
    # =========================================================================

    TEMP_BASE = "/tmp/airweave"
    TEMP_PROCESSING = f"{TEMP_BASE}/processing"
    TEMP_CACHE = f"{TEMP_BASE}/cache"

    ARF_PREFIX = "raw"

    CTTI_GLOBAL_DIR = "aactmarkdowns"

    # =========================================================================
    # ARF path builders
    # =========================================================================

    @classmethod
    def arf_sync_path(cls, sync_id: Union[str, UUID]) -> str:
        """Base path for a sync's ARF data: raw/{sync_id}/."""
        return f"{cls.ARF_PREFIX}/{sync_id}"

    @classmethod
    def arf_manifest_path(cls, sync_id: Union[str, UUID]) -> str:
        """Manifest path: raw/{sync_id}/manifest.json."""
        return f"{cls.arf_sync_path(sync_id)}/manifest.json"

    @classmethod
    def arf_entity_path(cls, sync_id: Union[str, UUID], entity_id: str) -> str:
        """Entity path: raw/{sync_id}/entities/{safe_entity_id}.json."""
        safe_id = cls.safe_filename(entity_id)
        return f"{cls.arf_sync_path(sync_id)}/entities/{safe_id}.json"

    @classmethod
    def arf_file_path(
        cls, sync_id: Union[str, UUID], entity_id: str, filename: Optional[str] = None
    ) -> str:
        """File path: raw/{sync_id}/files/{entity_id}_{name}.{ext}."""
        safe_id = cls.safe_filename(entity_id)
        if filename:
            name = Path(filename).stem
            ext = Path(filename).suffix or ""
            safe_name = cls.safe_filename(name)
            return f"{cls.arf_sync_path(sync_id)}/files/{safe_id}_{safe_name}{ext}"
        return f"{cls.arf_sync_path(sync_id)}/files/{safe_id}"

    @classmethod
    def arf_entities_dir(cls, sync_id: Union[str, UUID]) -> str:
        """Entities directory: raw/{sync_id}/entities/."""
        return f"{cls.arf_sync_path(sync_id)}/entities"

    @classmethod
    def arf_files_dir(cls, sync_id: Union[str, UUID]) -> str:
        """Files directory: raw/{sync_id}/files/."""
        return f"{cls.arf_sync_path(sync_id)}/files"

    # =========================================================================
    # Upload path builders (neo_file_upload source)
    # =========================================================================

    UPLOAD_PREFIX = "uploads"

    @classmethod
    def upload_prefix(cls, organization_id: Union[str, UUID], collection_readable_id: str) -> str:
        """Base prefix for a collection's uploaded files.

        uploads/{organization_id}/{collection_readable_id}/
        """
        return f"{cls.UPLOAD_PREFIX}/{organization_id}/{collection_readable_id}"

    @classmethod
    def upload_file_path(
        cls,
        organization_id: Union[str, UUID],
        collection_readable_id: str,
        relative_path: str,
    ) -> str:
        """Full path for one uploaded file under a collection's upload prefix.

        The caller is responsible for validating ``relative_path`` is a safe,
        relative, non-traversing path before calling this.
        """
        return f"{cls.upload_prefix(organization_id, collection_readable_id)}/{relative_path}"

    # =========================================================================
    # Temp path builders
    # =========================================================================

    @classmethod
    def temp_sync_dir(cls, sync_job_id: UUID) -> str:
        """Temp directory for a sync job: /tmp/airweave/processing/{sync_job_id}/."""
        return f"{cls.TEMP_PROCESSING}/{sync_job_id}"

    @classmethod
    def temp_file_path(cls, sync_job_id: UUID, file_uuid: str, filename: str) -> str:
        """Temp file path: /tmp/airweave/processing/{sync_job_id}/{uuid}-{name}."""
        safe_name = cls.safe_filename(filename)
        return f"{cls.temp_sync_dir(sync_job_id)}/{file_uuid}-{safe_name}"

    @classmethod
    def temp_cache_dir(cls) -> str:
        """Cache directory for downloaded files."""
        return cls.TEMP_CACHE

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def safe_filename(value: str, max_length: int = 200) -> str:
        """Convert value to safe storage path.

        Uses hash suffix for long/complex values to ensure uniqueness.
        """
        safe = _UNSAFE_CHARS_RE.sub("_", str(value))
        safe = _MULTI_UNDERSCORE_RE.sub("_", safe).strip("_")

        if len(safe) > max_length or safe != value:
            prefix = safe[:50] if len(safe) > 50 else safe
            hash_suffix = hashlib.md5(value.encode(), usedforsecurity=False).hexdigest()[:12]
            safe = f"{prefix}_{hash_suffix}"

        return safe[:max_length]

    _safe_filename = safe_filename

    @staticmethod
    def safe_relative_path(value: str, max_segment_length: int = 200) -> Optional[str]:
        """Validate and normalize a client-supplied relative file path.

        Unlike :meth:`safe_filename`, this preserves folder structure (each
        ``/``-separated segment is sanitized individually) rather than
        collapsing the whole value to one hashed name — needed so an
        uploaded folder's structure survives.

        Returns ``None`` if the path is empty, absolute, contains a ``..``
        traversal segment, or contains a drive letter — callers MUST treat
        ``None`` as "reject this file", never fall back to a default path.
        """
        if not value:
            return None

        # Reject Windows drive letters (e.g. "C:\\Windows") specifically —
        # a single ASCII letter followed by ':' at the very start — rather
        # than any colon anywhere, so a colon inside an ordinary filename is
        # sanitized like any other unsafe character instead of rejected.
        if _DRIVE_LETTER_RE.match(value):
            return None

        posix_value = value.replace("\\", "/")
        if posix_value.startswith("/"):
            return None

        segments = [seg for seg in posix_value.split("/") if seg not in ("", ".")]
        if not segments or any(seg == ".." for seg in segments):
            return None

        safe_segments = [
            _MULTI_UNDERSCORE_RE.sub("_", _UNSAFE_CHARS_RE.sub("_", seg)).strip("_")[
                :max_segment_length
            ]
            for seg in segments
        ]
        if any(not seg for seg in safe_segments):
            return None

        return "/".join(safe_segments)


# Convenience alias
paths = StoragePaths
