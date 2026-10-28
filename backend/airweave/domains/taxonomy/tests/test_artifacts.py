"""Per-sync taxonomy resolution from S3 (plan A7 artifacts.py; contract section 8, C1, C17, C20).

The bucket is a ``FilesystemBackend`` laid out as Meridian writes it. Version files here are
stand-in bytes: the resolver only verifies and downloads them.
"""

import hashlib
import json
from pathlib import Path

import pytest

from airweave.adapters.storage.filesystem import FilesystemBackend
from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError, TaxonomyResolver
from airweave.domains.taxonomy.tests.taxonomy_fixtures import activate, publish

TENANT, VERSION, READABLE = "t-acme", "acme-2026.10", "neo-acme-1"


class CountingBackend(FilesystemBackend):
    def __init__(self, base_path):
        super().__init__(base_path)
        self.reads: list = []

    async def read_file(self, path: str) -> bytes:
        self.reads.append(path)
        return await super().read_file(path)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _fake_build(tmp: Path) -> tuple[Path, dict]:
    built = tmp / "build"
    built.mkdir(parents=True, exist_ok=True)
    final = {
        "categories": [
            {"category_id": "CAT001", "name": "Loan Quality", "broad_family_id": "BF01"},
            {"category_id": "CAT002", "name": "Revenue", "broad_family_id": "BF01"},
        ],
        "broad_families": [{"broad_family_id": "BF01", "name": "Credit"}],
    }
    (built / "final_taxonomy.json").write_text(json.dumps(final))
    for n in ("dense_prototypes.npy", "sparse_profiles.npz", "bm25_vocabulary.json"):
        (built / n).write_bytes(f"stand-in {n}".encode())
    cfg = {"embedding": {"model": "fake-bow", "dimensions": 64}, "assignment": {}}
    return built, cfg


@pytest.fixture
def s3(tmp_path):
    return tmp_path / "s3"


@pytest.fixture
def published(tmp_path, s3):
    built, cfg = _fake_build(tmp_path)
    return publish(s3, built, cfg, TENANT, VERSION, readable_ids=[READABLE])


def _resolver(tmp_path, s3, clock=None, ttl=300):
    backend = CountingBackend(s3)
    r = TaxonomyResolver(backend, "taxonomy/", tmp_path / "local", ttl, clock or Clock())
    return r, backend


def _version_file(s3, name):
    return s3 / f"taxonomy/tenants/{TENANT}/versions/{VERSION}" / name


class TestResolve:
    async def test_no_active_file_is_not_an_error(self, tmp_path, s3):
        resolver, backend = _resolver(tmp_path, s3)
        assert await resolver.resolve("neo-unknown") is None
        assert await resolver.resolve("neo-unknown") is None
        assert len(backend.reads) == 1  # the answer is cached

    async def test_resolves_and_verifies(self, tmp_path, s3, published):
        resolver, _ = _resolver(tmp_path, s3)
        t = await resolver.resolve(READABLE)
        assert (t.tenant_id, t.taxonomy_version, t.manifest_sha256) == (
            TENANT,
            VERSION,
            published.manifest_sha256,
        )
        assert t.acceptance_policy == "calibrated" and t.acceptance_model_id is None
        assert t.cache_key == (VERSION, published.manifest_sha256)
        assert (
            t.local_path == tmp_path / "local" / TENANT / f"{VERSION}-{published.manifest_sha256}"
        )
        assert (t.local_path / "taxonomy.yaml").is_file()
        assert t.registry.categories == {
            "CAT001": "t-acme:credit.loan_quality",
            "CAT002": "t-acme:credit.revenue",
        }
        assert t.generation == 1

    async def test_cached_until_ttl(self, tmp_path, s3, published):
        clock = Clock()
        resolver, backend = _resolver(tmp_path, s3, clock)
        await resolver.resolve(READABLE)
        reads = len(backend.reads)
        clock.now += 299
        await resolver.resolve(READABLE)
        assert len(backend.reads) == reads
        clock.now += 2
        await resolver.resolve(READABLE)
        assert len(backend.reads) == reads + 1  # the active file only; the version is cached

    async def test_switch_version_on_next_resolve_after_ttl(self, tmp_path, s3, published):
        clock = Clock()
        resolver, _ = _resolver(tmp_path, s3, clock)
        assert (await resolver.resolve(READABLE)).taxonomy_version == VERSION
        built, cfg = _fake_build(tmp_path / "v2")
        p2 = publish(s3, built, cfg, TENANT, "acme-2026.11")
        activate(s3, READABLE, TENANT, "acme-2026.11", p2.manifest_sha256, generation=2)
        assert (await resolver.resolve(READABLE)).taxonomy_version == VERSION  # within the TTL
        clock.now += 301
        t = await resolver.resolve(READABLE)
        assert (t.taxonomy_version, t.generation) == ("acme-2026.11", 2)

    async def test_reuploaded_version_name_is_a_new_folder(self, tmp_path, s3, published):
        """Same version name, other content: the manifest checksum keeps them apart (C1)."""
        clock = Clock()
        resolver, _ = _resolver(tmp_path, s3, clock)
        first = await resolver.resolve(READABLE)
        built, cfg = _fake_build(tmp_path / "again")
        (built / "bm25_vocabulary.json").write_bytes(b"changed")
        again = publish(s3, built, cfg, TENANT, VERSION, readable_ids=[READABLE])
        clock.now += 301
        second = await resolver.resolve(READABLE)
        assert second.manifest_sha256 == again.manifest_sha256 != first.manifest_sha256
        assert second.local_path != first.local_path
        assert (second.local_path / "bm25_vocabulary.json").read_bytes() == b"changed"


class TestRefused:
    async def _refused(self, tmp_path, s3, match):
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match=match):
            await resolver.resolve(READABLE)

    async def test_active_file_naming_another_collection(self, tmp_path, s3, published):
        active = s3 / f"taxonomy/active/{READABLE}.json"
        data = json.loads(active.read_text())
        data["collection_readable_id"] = "neo-globex-1"
        active.write_text(json.dumps(data))
        await self._refused(tmp_path, s3, "names collection")

    async def test_version_folder_of_another_tenant(self, tmp_path, s3, published):
        active = s3 / f"taxonomy/active/{READABLE}.json"
        data = json.loads(active.read_text())
        data["tenant_id"] = "t-globex"
        active.write_text(json.dumps(data))
        await self._refused(tmp_path, s3, "not the tenant's version folder")

    async def test_unsafe_names(self, tmp_path, s3, published):
        active = s3 / f"taxonomy/active/{READABLE}.json"
        data = json.loads(active.read_text())
        data["taxonomy_version"] = "../../etc"
        active.write_text(json.dumps(data))
        await self._refused(tmp_path, s3, "unsafe")

    async def test_manifest_checksum_mismatch(self, tmp_path, s3, published):
        activate(s3, READABLE, TENANT, VERSION, "0" * 64)
        await self._refused(tmp_path, s3, "does not match manifest_sha256")

    async def test_file_checksum_mismatch(self, tmp_path, s3, published):
        _version_file(s3, "dense_prototypes.npy").write_bytes(b"tampered")
        await self._refused(tmp_path, s3, "does not match manifest.json")

    async def test_required_file_not_listed(self, tmp_path, s3, published):
        mf = _version_file(s3, "manifest.json")
        man = json.loads(mf.read_text())
        man["files"] = [f for f in man["files"] if f["file"] != "taxonomy.yaml"]
        raw = json.dumps(man).encode()
        mf.write_bytes(raw)
        activate(s3, READABLE, TENANT, VERSION, hashlib.sha256(raw).hexdigest())
        await self._refused(tmp_path, s3, "does not list")

    async def test_policy_disagreement(self, tmp_path, s3, published):
        mf = _version_file(s3, "manifest.json")
        man = json.loads(mf.read_text())
        man["acceptance_policy"] = "learned"
        raw = json.dumps(man).encode()
        mf.write_bytes(raw)
        activate(s3, READABLE, TENANT, VERSION, hashlib.sha256(raw).hexdigest())
        await self._refused(tmp_path, s3, "taxonomy.yaml says 'calibrated'")

    async def test_no_bucket(self, tmp_path):
        resolver = TaxonomyResolver(None, "taxonomy/", tmp_path, 300)
        with pytest.raises(TaxonomyResolutionError, match="TAXONOMY_S3_BUCKET"):
            await resolver.resolve(READABLE)

    async def test_failure_cached_then_recovers(self, tmp_path, s3, published):
        clock = Clock()
        resolver, _ = _resolver(tmp_path, s3, clock)
        good = (s3 / f"taxonomy/active/{READABLE}.json").read_text()
        activate(s3, READABLE, TENANT, VERSION, "0" * 64)
        for _ in range(2):
            with pytest.raises(TaxonomyResolutionError):
                await resolver.resolve(READABLE)
        (s3 / f"taxonomy/active/{READABLE}.json").write_text(good)
        clock.now += 301
        assert (await resolver.resolve(READABLE)).taxonomy_version == VERSION


class TestLearnedPolicyWithoutPackage:
    """Model checks that fail before the package is needed."""

    def _learned(self, tmp_path, s3, model_dir=None):
        built, cfg = _fake_build(tmp_path)
        return publish(
            s3,
            built,
            cfg,
            TENANT,
            VERSION,
            readable_ids=[READABLE],
            policy="learned",
            model_dir=model_dir,
            model_id="acme_hgb",
        )

    async def test_manifest_without_model_sha256(self, tmp_path, s3):
        self._learned(tmp_path, s3)
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="acceptance_model_sha256"):
            await resolver.resolve(READABLE)

    async def test_model_manifest_missing(self, tmp_path, s3):
        model = tmp_path / "model"
        model.mkdir()
        (model / "frozen_model.skops").write_bytes(b"not a model")
        (model / "manifest.json").write_text("{}")
        self._learned(tmp_path, s3, model)
        (s3 / "taxonomy/models/acme_hgb/manifest.json").unlink()
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="manifest.json is missing"):
            await resolver.resolve(READABLE)

    async def test_model_manifest_names_another_file_hash(self, tmp_path, s3):
        model = tmp_path / "model"
        model.mkdir()
        (model / "frozen_model.skops").write_bytes(b"not a model")
        (model / "manifest.json").write_text(json.dumps({"model_sha256": "1" * 64}))
        self._learned(tmp_path, s3, model)
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="model_sha256 differs"):
            await resolver.resolve(READABLE)


class TestLearnedPolicyWithPackage:
    """Model checks that need the classifier package (its feature schema, scikit-learn)."""

    @pytest.fixture(autouse=True)
    def _package(self):
        pytest.importorskip("taxonomy_classifier")
        pytest.importorskip("sklearn")

    def _learned(self, tmp_path, s3, **model_kw):
        from airweave.domains.taxonomy.tests.taxonomy_fixtures import write_model

        write_model(tmp_path / "model", 0.9, **model_kw)
        built, cfg = _fake_build(tmp_path)
        publish(
            s3,
            built,
            cfg,
            TENANT,
            VERSION,
            readable_ids=[READABLE],
            policy="learned",
            model_dir=tmp_path / "model",
            model_id="acme_hgb",
        )

    async def test_verified_model_wired_into_config(self, tmp_path, s3):
        self._learned(tmp_path, s3)
        resolver, _ = _resolver(tmp_path, s3)
        t = await resolver.resolve(READABLE)
        acc = t.config["assignment"]["acceptance"]
        assert (t.acceptance_policy, t.acceptance_model_id) == ("learned", "acme_hgb")
        assert acc["policy"] == "learned"
        assert (
            Path(acc["model_path"]).read_bytes()
            == (tmp_path / "model/frozen_model.skops").read_bytes()
        )
        assert Path(acc["model_manifest"]).is_file()

    async def test_model_file_missing(self, tmp_path, s3):
        self._learned(tmp_path, s3)
        (s3 / "taxonomy/models/acme_hgb/frozen_model.skops").unlink()
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="frozen_model.skops is missing"):
            await resolver.resolve(READABLE)

    async def test_model_file_tampered(self, tmp_path, s3):
        self._learned(tmp_path, s3)
        (s3 / "taxonomy/models/acme_hgb/frozen_model.skops").write_bytes(b"swapped model")
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="does not match its sha256"):
            await resolver.resolve(READABLE)
        assert not (tmp_path / "local/models").exists() or not any(
            (tmp_path / "local/models").rglob("*.skops")
        )

    async def test_other_feature_schema(self, tmp_path, s3):
        self._learned(tmp_path, s3, schema="learned-acceptance-features/0")
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="feature schema"):
            await resolver.resolve(READABLE)


class TestPathComponents:
    """A tenant or version is one plain path component: never ``.``, ``..`` or a path."""

    @pytest.mark.parametrize(
        "name", [".", "..", "../x", "a/b", "a\\b", "", ".hidden", "-lead", "a b", "x/.."]
    )
    def test_unsafe_names_are_refused(self, name):
        from airweave.domains.taxonomy.artifacts import is_safe_name

        assert is_safe_name(name) is False

    @pytest.mark.parametrize("name", ["t-acme", "acme-2026.10", "v1", "a.b_c+d-e", "T9"])
    def test_plain_names_are_accepted(self, name):
        from airweave.domains.taxonomy.artifacts import is_safe_name

        assert is_safe_name(name) is True

    @pytest.mark.parametrize("tenant", [".", ".."])
    async def test_a_dot_tenant_in_an_active_file_never_reaches_the_filesystem(
        self, tmp_path, s3, published, tenant
    ):
        active = s3 / f"taxonomy/active/{READABLE}.json"
        data = json.loads(active.read_text())
        data["tenant_id"] = tenant
        active.write_text(json.dumps(data))
        resolver, backend = _resolver(tmp_path, s3)

        with pytest.raises(TaxonomyResolutionError, match="unsafe"):
            await resolver.resolve(READABLE)

        assert not (tmp_path / "local").exists() or not any((tmp_path / "local").iterdir())
        assert not any("manifest.json" in r for r in backend.reads)

    async def test_a_local_dir_that_the_version_would_leave_is_refused(self, tmp_path, s3):
        resolver, _ = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="outside the taxonomy cache"):
            resolver._beneath_local_dir(tmp_path / "local" / ".." / "elsewhere")


class TestArtifactSizes:
    def _resolver_with(self, tmp_path, s3, **limits):
        return TaxonomyResolver(
            CountingBackend(s3), "taxonomy/", tmp_path / "local", 300, Clock(), **limits
        )

    def _declare(self, s3, name, size):
        mf = _version_file(s3, "manifest.json")
        man = json.loads(mf.read_text())
        for entry in man["files"]:
            if entry["file"] == name:
                entry["size"] = size
        raw = json.dumps(man).encode()
        mf.write_bytes(raw)
        activate(s3, READABLE, TENANT, VERSION, hashlib.sha256(raw).hexdigest())

    async def test_a_declared_size_over_the_file_limit_is_refused_before_any_download(
        self, tmp_path, s3, published
    ):
        self._declare(s3, "dense_prototypes.npy", 10_000)
        resolver = self._resolver_with(tmp_path, s3, max_file_bytes=1_000)

        with pytest.raises(TaxonomyResolutionError, match="declares 10000 bytes"):
            await resolver.resolve(READABLE)

        assert not (tmp_path / "local").exists() or not any((tmp_path / "local").rglob("*.npy"))
        assert not any(r.endswith("dense_prototypes.npy") for r in resolver._storage.reads)

    async def test_a_file_that_declares_nothing_is_checked_when_it_is_read(
        self, tmp_path, s3, published
    ):
        resolver = self._resolver_with(tmp_path, s3, max_file_bytes=10)

        with pytest.raises(TaxonomyResolutionError, match="over the limits"):
            await resolver.resolve(READABLE)

    async def test_the_total_is_bounded_too(self, tmp_path, s3, published):
        resolver = self._resolver_with(tmp_path, s3, max_file_bytes=10_000, max_total_bytes=50)

        with pytest.raises(TaxonomyResolutionError, match="over the limits|in all"):
            await resolver.resolve(READABLE)

    async def test_a_malformed_declared_size_is_refused(self, tmp_path, s3, published):
        self._declare(s3, "taxonomy.yaml", -5)
        resolver = self._resolver_with(tmp_path, s3)

        with pytest.raises(TaxonomyResolutionError, match="bad size"):
            await resolver.resolve(READABLE)

    async def test_the_defaults_leave_normal_artifacts_alone(self, tmp_path, s3, published):
        resolver = self._resolver_with(tmp_path, s3)
        assert (await resolver.resolve(READABLE)).taxonomy_version == VERSION
