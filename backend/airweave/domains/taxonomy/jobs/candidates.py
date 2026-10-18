"""Write-once candidate folders (P7 design, "Candidate write rule").

A discover or propose job writes only under
``taxonomy/tenants/<tenant>/candidates/<candidate_version>/``, never under ``versions/``, and
never over an existing object: every write is a conditional put (``If-None-Match: *``). The
first object is ``_claim.json`` naming the job, so a second job for the same candidate version
is refused before it spends anything; ``report.json`` goes last and marks the folder complete.

Replays are harmless: an object that already exists with the same bytes is accepted, so a job
that crashed half-way and runs again with the same files completes the folder. Different bytes
under an existing name are a conflict and fail the job.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol

from airweave.domains.storage.exceptions import StorageNotFoundError

CLAIM = "_claim.json"
REPORT = "report.json"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class CandidateTaken(Exception):
    """Another job already claimed this candidate folder."""


class CandidateConflict(Exception):
    """An object of the folder exists with other bytes than the ones to write."""


class OnceStorage(Protocol):
    """The two operations used: a conditional write and a read."""

    async def write_file_once(self, path: str, content: bytes) -> bool:
        """Write if absent; ``True`` if written, ``False`` if it exists."""
        ...

    async def read_file(self, path: str) -> bytes:
        """Read an object."""
        ...

    async def delete(self, path: str) -> bool:
        """Delete an object (the claim of a job that ended before it wrote anything)."""
        ...


def _safe(name: str, what: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"unsafe {what} {name!r}")
    return name


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CandidateWriter:
    """Writes one candidate folder for one job."""

    def __init__(self, storage: OnceStorage, prefix: str = "taxonomy/") -> None:
        """``prefix`` is ``TAXONOMY_S3_PREFIX``."""
        self._storage = storage
        self._prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"

    def folder(self, tenant_id: str, candidate_version: str) -> str:
        """``taxonomy/tenants/<tenant>/candidates/<version>/``."""
        return (
            f"{self._prefix}tenants/{_safe(tenant_id, 'tenant id')}/candidates/"
            f"{_safe(candidate_version, 'candidate version')}/"
        )

    async def claim(self, tenant_id: str, candidate_version: str, job_id: str, kind: str) -> None:
        """Take the folder for this job; raises ``CandidateTaken`` if another job has it."""
        path = self.folder(tenant_id, candidate_version) + CLAIM
        body = json.dumps({"job_id": job_id, "kind": kind}, sort_keys=True).encode()
        if await self._storage.write_file_once(path, body):
            return
        existing = json.loads(await self._storage.read_file(path))
        if existing.get("job_id") != job_id:
            raise CandidateTaken(
                f"candidate version {candidate_version!r} was already claimed by job "
                f"{existing.get('job_id')}"
            )

    async def release(self, tenant_id: str, candidate_version: str, job_id: str) -> None:
        """Give the folder up: delete this job's claim when nothing else was written to it.

        A job that ended before it built anything (no input, a cancel, too little text) leaves no
        candidate; its claim must not block the version name for ever. A claim of another job, or
        a folder that already holds objects, is left alone.
        """
        folder = self.folder(tenant_id, candidate_version)
        try:
            claim = json.loads(await self._storage.read_file(folder + CLAIM))
        except (StorageNotFoundError, ValueError):
            return
        if claim.get("job_id") != job_id:
            return
        try:
            await self._storage.delete(folder + CLAIM)
        except Exception:  # noqa: BLE001 - best effort; the claim only blocks a name
            return

    async def existing_report(
        self, tenant_id: str, candidate_version: str
    ) -> Optional[Dict[str, Any]]:
        """The folder's report if the folder is complete, else ``None``."""
        try:
            raw = await self._storage.read_file(self.folder(tenant_id, candidate_version) + REPORT)
        except StorageNotFoundError:
            return None
        report: Dict[str, Any] = json.loads(raw)
        return report

    async def publish(
        self,
        tenant_id: str,
        candidate_version: str,
        job_id: str,
        files: Mapping[str, bytes],
        report: Dict[str, Any],
        checkpoint: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Write the files, then the report. Returns the report as written (with checksums).

        ``checkpoint`` is awaited before every object, the report included; it raises to stop the
        publish (a cancel, a lost lease), so nothing more is written once it has been seen.
        """
        folder = self.folder(tenant_id, candidate_version)
        checksums = []
        for name in sorted(files):
            if checkpoint is not None:
                await checkpoint()
            await self._put(folder + _safe(name, "file name"), files[name])
            checksums.append({"file": name, "sha256": _sha(files[name])})
        final = {**report, "job_id": job_id, "files": checksums}
        if checkpoint is not None:
            await checkpoint()
        await self._put(folder + REPORT, json.dumps(final, indent=2, sort_keys=True).encode())
        return final

    async def _put(self, path: str, content: bytes) -> None:
        if await self._storage.write_file_once(path, content):
            return
        if _sha(await self._storage.read_file(path)) != _sha(content):
            raise CandidateConflict(f"{path} already exists with different content")
