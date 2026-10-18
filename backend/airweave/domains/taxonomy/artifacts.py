"""Resolve a collection's taxonomy for a sync (contract airweave-neo/1, section 8; C1, C17, C20).

Per sync, never from a process-wide setting: the collection's ``readable_id`` (from Airweave's own
database) names ``active/<readable_id>.json``, which names a version folder under the tenant's
prefix and that folder's ``manifest.json`` checksum. Everything is verified before use:

- the active file names the same collection, and its version folder lies under its tenant;
- ``manifest.json`` matches ``manifest_sha256``, and every file it lists matches its sha256;
- a version whose ``taxonomy.yaml`` names the learned policy gets its model from
  ``models/<acceptance_model_id>/``, and ``frozen_model.pkl`` must match both the version's
  ``acceptance_model_sha256`` and the model manifest before anything unpickles it.

Any failure raises :class:`TaxonomyResolutionError`: every chunk of the batch is then ``ERROR``,
and a learned-policy version is never classified with the calibrated policy instead (C20).
A missing active file is not an error: the collection has no taxonomy yet (``NOT_LABELLED``).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml  # type: ignore[import-untyped]

from airweave.domains.storage.exceptions import StorageNotFoundError
from airweave.domains.storage.protocols import StorageBackend
from airweave.domains.taxonomy.mapping import RegistryMap, RegistryMapError

CALIBRATED, LEARNED = "calibrated", "learned"

# The folder every taxonomy file lives under in the bucket: the same one Meridian writes. A constant
# in both repositories, so the two cannot disagree.
TAXONOMY_S3_PREFIX = "taxonomy/"

# Files a version folder must hold, each listed in manifest.json with its sha256 (contract 8)
REQUIRED_FILES = (
    "final_taxonomy.json",
    "dense_prototypes.npy",
    "sparse_profiles.npz",
    "bm25_vocabulary.json",
    "taxonomy.yaml",
    "registry_map.json",
)
DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class TaxonomyResolutionError(Exception):
    """The collection's taxonomy cannot be used: every chunk of the batch gets ``ERROR``."""


@dataclass(frozen=True)
class ResolvedTaxonomy:
    """A verified taxonomy version on local disk, ready for the classifier."""

    collection_readable_id: str
    tenant_id: str
    taxonomy_version: str
    manifest_sha256: str
    generation: int
    acceptance_policy: str
    acceptance_model_id: Optional[str]
    local_path: Path
    config: Dict[str, Any]  # taxonomy.yaml, with the verified model's local paths filled in
    registry: RegistryMap
    # From the active file (P7 design section 3). ``taxonomy_version`` above is the LABELLING
    # version; the serving version is only logged, never used to pick a classifier.
    serving_version: Optional[str] = None
    migration_id: Optional[str] = None

    @property
    def cache_key(self) -> Tuple[str, str]:
        """Loaded classifiers are cached by this (contract 8, "Classifier cache")."""
        return (self.taxonomy_version, self.manifest_sha256)


@dataclass(frozen=True)
class _Version:
    local_path: Path
    config: Dict[str, Any]
    registry: RegistryMap
    acceptance_policy: str
    acceptance_model_id: Optional[str]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_safe_name(name: Any) -> bool:
    """Whether ``name`` may be one path component: a plain name, never ``.``, ``..`` or a path.

    The pattern needs a first character that is a letter or digit, which already refuses ``.`` and
    ``..``; the explicit checks keep that true if the pattern is ever loosened.
    """
    return (
        isinstance(name, str)
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        and bool(_SAFE_NAME.match(name))
    )


def _safe(name: Any, what: str) -> str:
    if not is_safe_name(name):
        raise TaxonomyResolutionError(f"unsafe {what} {name!r}")
    return name


class TaxonomyResolver:
    """Reads active files and version folders from the taxonomy bucket, verified and cached."""

    def __init__(
        self,
        storage: Optional[StorageBackend],
        prefix: str,
        local_dir: str | Path,
        active_cache_seconds: int,
        clock: Callable[[], float] = time.monotonic,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        """Initialize with the bucket's storage (None: no bucket configured) and local paths.

        ``max_file_bytes`` and ``max_total_bytes`` bound one artifact file and one version's
        files together (a manifest that declares more is refused before any download).
        """
        self._max_file = max_file_bytes
        self._max_total = max_total_bytes
        self._storage = storage
        self._prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
        self._local_dir = Path(local_dir)
        self._ttl = active_cache_seconds
        self._clock = clock
        self._active: Dict[str, Tuple[float, Any]] = {}  # readable_id -> (expires, result)
        self._versions: Dict[Tuple[str, str, str], _Version] = {}
        self._models: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
        self._lock = asyncio.Lock()

    async def resolve(
        self, collection_readable_id: str, fresh: bool = False
    ) -> Optional[ResolvedTaxonomy]:
        """Return the collection's verified taxonomy, or None when it has no active file.

        The answer (including a failure) is cached for ``limits.ACTIVE_CACHE_SECONDS``, so a
        collection switches version, or recovers, on a sync after that. ``fresh`` skips the
        cached answer and reads the active file again (a re-label job starting right after the
        platform rewrote the file must not label with the previous version).
        """
        now = self._clock()
        cached = self._active.get(collection_readable_id)
        if not fresh and cached is not None and cached[0] > now:
            return self._unwrap(cached[1])
        async with self._lock:
            cached = self._active.get(collection_readable_id)
            if not fresh and cached is not None and cached[0] > self._clock():
                return self._unwrap(cached[1])
            try:
                result: Any = await self._resolve(collection_readable_id)
            except TaxonomyResolutionError as e:
                result = e
            except Exception as e:  # storage, parsing, disk: all mean the version is unusable
                result = TaxonomyResolutionError(f"{type(e).__name__}: {e}")
            self._active[collection_readable_id] = (self._clock() + self._ttl, result)
        return self._unwrap(result)

    @staticmethod
    def _unwrap(result: Any) -> Optional[ResolvedTaxonomy]:
        if isinstance(result, TaxonomyResolutionError):
            raise result
        return result if isinstance(result, ResolvedTaxonomy) else None

    async def _resolve(self, readable_id: str) -> Optional[ResolvedTaxonomy]:
        if self._storage is None:
            raise TaxonomyResolutionError("TAXONOMY_S3_BUCKET is not set")
        try:
            raw = await self._storage.read_file(f"{self._prefix}active/{readable_id}.json")
        except StorageNotFoundError:
            return None
        active = json.loads(raw)
        if active.get("collection_readable_id") != readable_id:
            raise TaxonomyResolutionError(
                f"active file for {readable_id!r} names collection "
                f"{active.get('collection_readable_id')!r}"
            )
        tenant = _safe(active.get("tenant_id"), "tenant_id")
        version = self._labelling_version(active)
        serving = active.get("serving_version")
        serving = _safe(serving, "serving_version") if serving is not None else None
        migration_id = active.get("migration_id")
        migration_id = _safe(migration_id, "migration_id") if migration_id is not None else None
        manifest_sha = active.get("manifest_sha256")
        if not isinstance(manifest_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
            raise TaxonomyResolutionError(f"active file for {readable_id!r}: bad manifest_sha256")
        prefix = active.get("version_prefix") or ""
        expected = f"{self._prefix}tenants/{tenant}/versions/{version}/"
        if prefix != expected:
            raise TaxonomyResolutionError(
                f"version_prefix {prefix!r} is not the tenant's version folder {expected!r}"
            )
        key = (tenant, version, manifest_sha)
        if key not in self._versions:
            self._versions[key] = await self._load_version(tenant, version, manifest_sha, prefix)
        v = self._versions[key]
        return ResolvedTaxonomy(
            collection_readable_id=readable_id,
            tenant_id=tenant,
            taxonomy_version=version,
            manifest_sha256=manifest_sha,
            generation=int(active.get("generation") or 0),
            acceptance_policy=v.acceptance_policy,
            acceptance_model_id=v.acceptance_model_id,
            local_path=v.local_path,
            config=v.config,
            registry=v.registry,
            serving_version=serving,
            migration_id=migration_id,
        )

    @staticmethod
    def _labelling_version(active: Dict[str, Any]) -> str:
        """The version new and changed chunks are labelled with (P7 design section 3).

        ``labelling_version`` when the file has it; an older file has only ``taxonomy_version``,
        which then is the labelling version. The platform writes both equal, so a disagreement
        is refused rather than guessed.
        """
        legacy = active.get("taxonomy_version")
        labelling = active.get("labelling_version")
        if labelling is None:
            return _safe(legacy, "taxonomy_version")
        labelling = _safe(labelling, "labelling_version")
        if legacy is not None and legacy != labelling:
            raise TaxonomyResolutionError(
                f"active file: taxonomy_version {legacy!r} and labelling_version "
                f"{labelling!r} disagree"
            )
        return labelling

    async def _load_version(
        self, tenant: str, version: str, manifest_sha: str, prefix: str
    ) -> _Version:
        raw = await self._storage.read_file(f"{prefix}manifest.json")  # type: ignore[union-attr]
        if _sha256(raw) != manifest_sha:
            raise TaxonomyResolutionError(f"{prefix}manifest.json does not match manifest_sha256")
        manifest = json.loads(raw)
        if manifest.get("taxonomy_version", version) != version:
            raise TaxonomyResolutionError(f"{prefix}manifest.json names another version")
        files = {_safe(f["file"], "file name"): f["sha256"] for f in manifest.get("files") or []}
        missing = [f for f in REQUIRED_FILES if f not in files]
        if missing:
            raise TaxonomyResolutionError(f"{prefix}manifest.json does not list {missing}")
        self._check_declared_sizes(prefix, manifest.get("files") or [])

        local = self._local_dir / tenant / f"{version}-{manifest_sha}"
        self._beneath_local_dir(local)
        await self._download(prefix, files, local)

        config = yaml.safe_load((local / "taxonomy.yaml").read_text()) or {}
        try:
            registry = RegistryMap.from_json(json.loads((local / "registry_map.json").read_text()))
        except RegistryMapError as e:
            raise TaxonomyResolutionError(str(e)) from e
        if registry.taxonomy_version != version:
            raise TaxonomyResolutionError("registry_map.json names another version")

        acceptance = (config.get("assignment") or {}).get("acceptance") or {}
        policy = acceptance.get("policy", CALIBRATED)
        if policy not in (CALIBRATED, LEARNED):
            raise TaxonomyResolutionError(f"unknown acceptance policy {policy!r}")
        if manifest.get("acceptance_policy", policy) != policy:
            raise TaxonomyResolutionError(
                f"manifest.json says {manifest.get('acceptance_policy')!r}, "
                f"taxonomy.yaml says {policy!r}"
            )
        model_id = None
        if policy == LEARNED:
            model_id = _safe(manifest.get("acceptance_model_id"), "acceptance_model_id")
            model_manifest, model_file = await self._model(
                model_id, manifest.get("acceptance_model_sha256")
            )
            config = copy.deepcopy(config)
            acc = config.setdefault("assignment", {}).setdefault("acceptance", {})
            acc["model_manifest"], acc["model_path"] = str(model_manifest), str(model_file)
        return _Version(local, config, registry, policy, model_id)

    def _beneath_local_dir(self, path: Path) -> None:
        """Refuse a version folder that resolves outside the taxonomy cache root."""
        root = self._local_dir.resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise TaxonomyResolutionError(f"{path} is outside the taxonomy cache {root}") from None

    def _check_declared_sizes(self, prefix: str, entries: Any) -> None:
        """Refuse a manifest whose declared sizes are over the limits, before downloading anything.

        A file that declares no size is checked after it is read; the storage interface has no
        way to ask for a size first.
        """
        total = 0
        for entry in entries:
            size = entry.get("size") if isinstance(entry, dict) else None
            if size is None:
                continue
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise TaxonomyResolutionError(
                    f"{prefix}manifest.json: bad size for {entry.get('file')!r}"
                )
            if size > self._max_file:
                raise TaxonomyResolutionError(
                    f"{prefix}{entry.get('file')} declares {size} bytes, "
                    f"over the {self._max_file} limit"
                )
            total += size
        if total > self._max_total:
            raise TaxonomyResolutionError(
                f"{prefix}manifest.json declares {total} bytes in all, "
                f"over the {self._max_total} limit"
            )

    async def _download(self, prefix: str, files: Dict[str, str], local: Path) -> None:
        """Download and verify every listed file; a verified folder on disk is reused."""
        if local.is_dir() and all(
            (local / f).is_file() and _sha256((local / f).read_bytes()) == h
            for f, h in files.items()
        ):
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{local.name}-", dir=local.parent))
        try:
            total = 0
            for name, expected in files.items():
                data = await self._storage.read_file(f"{prefix}{name}")  # type: ignore[union-attr]
                total += len(data)
                if len(data) > self._max_file or total > self._max_total:
                    raise TaxonomyResolutionError(
                        f"{prefix}{name}: {len(data)} bytes ({total} in all) are over the limits "
                        f"of {self._max_file} a file and {self._max_total} a version"
                    )
                if _sha256(data) != expected:
                    raise TaxonomyResolutionError(f"{prefix}{name} does not match manifest.json")
                (tmp / name).write_bytes(data)
            if local.exists():
                shutil.rmtree(local)
            os.replace(tmp, local)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    async def _model(self, model_id: str, expected_sha: Any) -> Tuple[Path, Path]:
        """Download ``models/<model_id>/`` once per process, verified before any unpickling."""
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise TaxonomyResolutionError(
                "learned policy, but manifest.json has no acceptance_model_sha256"
            )
        key = (model_id, expected_sha)
        if key in self._models:
            return self._models[key]
        base = f"{self._prefix}models/{model_id}/"
        try:
            man_raw = await self._storage.read_file(f"{base}manifest.json")  # type: ignore[union-attr]
        except StorageNotFoundError:
            raise TaxonomyResolutionError(f"learned policy, but {base}manifest.json is missing")
        manifest = json.loads(man_raw)
        model_name = _safe(manifest.get("model_file", "frozen_model.pkl"), "model_file")
        if manifest.get("model_sha256") != expected_sha:
            raise TaxonomyResolutionError(
                f"model {model_id}: its manifest's model_sha256 differs from the version's"
            )
        _check_feature_schema(model_id, manifest)
        try:
            data = await self._storage.read_file(f"{base}{model_name}")  # type: ignore[union-attr]
        except StorageNotFoundError:
            raise TaxonomyResolutionError(f"learned policy, but {base}{model_name} is missing")
        if len(data) > self._max_file:
            raise TaxonomyResolutionError(
                f"{base}{model_name}: {len(data)} bytes are over the {self._max_file} limit"
            )
        if _sha256(data) != expected_sha:
            raise TaxonomyResolutionError(f"{base}{model_name} does not match its sha256")
        local = self._local_dir / "models" / f"{model_id}-{expected_sha}"
        local.mkdir(parents=True, exist_ok=True)
        (local / "manifest.json").write_bytes(man_raw)
        (local / model_name).write_bytes(data)
        self._models[key] = (local / "manifest.json", local / model_name)
        return self._models[key]


def _check_feature_schema(model_id: str, manifest: Dict[str, Any]) -> None:
    """Refuse a model built for another feature schema, before it is ever loaded."""
    try:
        from taxonomy_classifier.assignment import learned  # type: ignore[import-not-found]
    except ImportError as e:
        raise TaxonomyResolutionError(f"the classifier package is not installed: {e}") from e
    if manifest.get("feature_schema_version") != learned.FEATURE_SCHEMA_VERSION or tuple(
        manifest.get("feature_names") or ()
    ) != tuple(learned.FEATURE_NAMES):
        raise TaxonomyResolutionError(
            f"model {model_id}: feature schema {manifest.get('feature_schema_version')!r} "
            f"is not the classifier's {learned.FEATURE_SCHEMA_VERSION!r}"
        )
