"""A version folder written exactly as Meridian writes it is read by the resolver (batch 3, item 4).

``meridian_publish`` repeats ``TaxonomyArtifactStore.publishVersion`` and ``writeActiveFile`` in
``neo-meridian-mesh-svc/src/taxonomy-registry/artifact-store.ts`` (platform commit d18203f79),
read only: the same keys in the same order, ``JSON.stringify(x, null, 2)`` for the version files
and compact ``JSON.stringify`` for the active file, ``null`` for the model keys of a calibrated
version, milliseconds in ``created_at``, and the manifest's checksum taken over its exact bytes.
The fork's other tests write manifests with Python helpers; this one is the shape cross-check.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from airweave.adapters.storage.filesystem import FilesystemBackend
from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError, TaxonomyResolver

TENANT, VERSION, READABLE = "t-3f9a21c0", "acme-2026.10", "neo-acme-7d41"
BUILD_FILES = {
    "final_taxonomy.json": b'{"categories": []}',
    "dense_prototypes.npy": b"stand-in prototypes",
    "sparse_profiles.npz": b"stand-in profiles",
    "bm25_vocabulary.json": b'["loan", "revenue"]',
    "taxonomy.yaml": b"assignment:\n  acceptance:\n    policy: {policy}\n",
}


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def js_json(value, indent=None) -> bytes:
    """What ``JSON.stringify`` writes: no ASCII escaping, ``indent`` spaces or compact."""
    if indent is None:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return json.dumps(value, indent=indent, ensure_ascii=False).encode()


def iso_ms(moment: datetime) -> str:
    """``new Date(x).toISOString()``."""
    utc = moment.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def meridian_publish(
    root: Path, policy: str, model_id=None, model_sha=None, created=None, generation=1
) -> str:
    """Write a version folder, registry map, manifest and active file; return manifest sha256."""
    created = created or datetime(2026, 10, 14, 9, 0, 0, 123000, tzinfo=timezone.utc)
    prefix = f"taxonomy/tenants/{TENANT}/versions/{VERSION}/"
    folder = root / prefix
    folder.mkdir(parents=True)
    checksums = []
    for name, body in BUILD_FILES.items():  # uploadBuildFiles
        body = body.replace(b"{policy}", policy.encode())
        (folder / name).write_bytes(body)
        checksums.append({"file": name, "sha256": sha(body)})
    registry_map = js_json(
        {
            "taxonomy_version": VERSION,
            "categories": {"CAT001": "acme:credit.loan_quality", "CAT002": "acme:credit.revenue"},
            "families": {"BF01": "acme:credit"},
        },
        2,
    )
    (folder / "registry_map.json").write_bytes(registry_map)
    checksums.append({"file": "registry_map.json", "sha256": sha(registry_map)})
    manifest = js_json(
        {
            "taxonomy_version": VERSION,
            "files": checksums,
            "classifier_version": "1.0.0",
            "embedding_model": "text-embedding-3-small",
            "acceptance_policy": policy,
            "acceptance_model_id": model_id or None,
            "acceptance_model_sha256": model_sha or None,
            "created_at": iso_ms(created),
        },
        2,
    )
    (folder / "manifest.json").write_bytes(manifest)
    active = root / "taxonomy" / "active"  # writeActiveFile
    active.mkdir(parents=True, exist_ok=True)
    (active / f"{READABLE}.json").write_bytes(
        js_json(
            {
                "collection_readable_id": READABLE,
                "tenant_id": TENANT,
                "taxonomy_version": VERSION,
                "version_prefix": prefix,
                "manifest_sha256": sha(manifest),
                "generation": generation,
                "updated_at": iso_ms(created),
            }
        )
    )
    return sha(manifest)


def resolver(tmp_path: Path) -> TaxonomyResolver:
    return TaxonomyResolver(
        FilesystemBackend(tmp_path / "s3"), "taxonomy/", tmp_path / "local", 300
    )


class TestShape:
    def test_the_bytes_have_the_platforms_form(self, tmp_path):
        sha_hex = meridian_publish(tmp_path / "s3", "calibrated")
        folder = tmp_path / "s3/taxonomy/tenants" / TENANT / "versions" / VERSION
        text = (folder / "manifest.json").read_text()
        assert text.startswith('{\n  "taxonomy_version": "acme-2026.10",\n  "files": [\n    {\n')
        keys = list(json.loads(text))
        assert keys == [
            "taxonomy_version",
            "files",
            "classifier_version",
            "embedding_model",
            "acceptance_policy",
            "acceptance_model_id",
            "acceptance_model_sha256",
            "created_at",
        ]
        assert '"acceptance_model_id": null' in text and "2026-10-14T09:00:00.123Z" in text
        assert sha(text.encode()) == sha_hex
        active = (tmp_path / "s3/taxonomy/active" / f"{READABLE}.json").read_text()
        assert " " not in active.replace("T09:00:00", "")  # compact, as JSON.stringify writes it


class TestCalibrated:
    async def test_read_correctly(self, tmp_path):
        manifest_sha = meridian_publish(tmp_path / "s3", "calibrated")
        t = await resolver(tmp_path).resolve(READABLE)
        assert (t.tenant_id, t.taxonomy_version, t.manifest_sha256) == (
            TENANT,
            VERSION,
            manifest_sha,
        )
        assert (t.acceptance_policy, t.acceptance_model_id, t.generation) == ("calibrated", None, 1)
        assert t.registry.categories["CAT002"] == "acme:credit.revenue"
        assert t.registry.families == {"BF01": "acme:credit"}
        assert (t.local_path / "taxonomy.yaml").read_bytes().startswith(b"assignment:")
        assert t.config["assignment"]["acceptance"]["policy"] == "calibrated"

    async def test_null_model_keys_are_not_a_model(self, tmp_path):
        meridian_publish(tmp_path / "s3", "calibrated", model_id="", model_sha="")
        assert (await resolver(tmp_path).resolve(READABLE)).acceptance_model_id is None

    async def test_a_changed_file_is_refused(self, tmp_path):
        meridian_publish(tmp_path / "s3", "calibrated")
        folder = tmp_path / "s3/taxonomy/tenants" / TENANT / "versions" / VERSION
        (folder / "final_taxonomy.json").write_bytes(b"{}")
        with pytest.raises(TaxonomyResolutionError):
            await resolver(tmp_path).resolve(READABLE)


class TestLearned:
    @pytest.fixture
    def model(self, tmp_path):
        pytest.importorskip("taxonomy_classifier")
        pytest.importorskip("sklearn")
        from airweave.domains.taxonomy.tests.taxonomy_fixtures import write_model

        source = tmp_path / "model-src"
        write_model(source, 0.9, model_id="sec_b_hgb_d3")
        dest = tmp_path / "s3/taxonomy/models/sec_b_hgb_d3"  # operator-managed, outside versions
        dest.mkdir(parents=True)
        for name in ("manifest.json", "frozen_model.skops"):
            shutil.copy(source / name, dest / name)
        return sha((source / "frozen_model.skops").read_bytes())

    async def test_read_correctly(self, tmp_path, model):
        manifest_sha = meridian_publish(tmp_path / "s3", "learned", "sec_b_hgb_d3", model)
        t = await resolver(tmp_path).resolve(READABLE)
        assert (t.acceptance_policy, t.acceptance_model_id) == ("learned", "sec_b_hgb_d3")
        assert t.manifest_sha256 == manifest_sha
        acceptance = t.config["assignment"]["acceptance"]
        assert acceptance["policy"] == "learned"
        assert (
            Path(acceptance["model_path"]).read_bytes()
            and Path(acceptance["model_manifest"]).is_file()
        )

    async def test_model_key_missing_from_the_manifest_is_an_error(self, tmp_path, model):
        meridian_publish(tmp_path / "s3", "learned", None, None)
        with pytest.raises(TaxonomyResolutionError):
            await resolver(tmp_path).resolve(READABLE)

    async def test_model_sha_that_differs_is_an_error(self, tmp_path, model):
        meridian_publish(tmp_path / "s3", "learned", "sec_b_hgb_d3", "0" * 64)
        with pytest.raises(TaxonomyResolutionError):
            await resolver(tmp_path).resolve(READABLE)

    async def test_policy_disagreement_is_an_error(self, tmp_path, model):
        meridian_publish(tmp_path / "s3", "calibrated")
        folder = tmp_path / "s3/taxonomy/tenants" / TENANT / "versions" / VERSION
        # Meridian says calibrated, taxonomy.yaml says learned, with the checksums re-sealed
        (folder / "taxonomy.yaml").write_bytes(b"assignment:\n  acceptance:\n    policy: learned\n")
        manifest = json.loads((folder / "manifest.json").read_text())
        for entry in manifest["files"]:
            entry["sha256"] = sha((folder / entry["file"]).read_bytes())
        raw = js_json(manifest, 2)
        (folder / "manifest.json").write_bytes(raw)
        active_path = tmp_path / "s3/taxonomy/active" / f"{READABLE}.json"
        active = json.loads(active_path.read_text())
        active["manifest_sha256"] = sha(raw)
        active_path.write_bytes(js_json(active))
        with pytest.raises(TaxonomyResolutionError, match="calibrated"):
            await resolver(tmp_path).resolve(READABLE)
