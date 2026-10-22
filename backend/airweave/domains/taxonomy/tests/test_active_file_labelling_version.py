"""The active file's new fields (P7 design section 3; batch 5, task 1).

Airweave labels with ``labelling_version`` and ignores ``serving_version``. During a migration
the file names the target as the labelling version while serving stays on the source; the
folder and checksum in the file belong to the labelling version (the platform's
``writeActiveFile``). An older file without the new fields keeps working.
"""

import json
import logging
from pathlib import Path

import pytest

from airweave.adapters.storage.filesystem import FilesystemBackend
from airweave.domains.taxonomy.artifacts import TaxonomyResolutionError, TaxonomyResolver
from airweave.domains.taxonomy.metrics import TaxonomyTally, log_sync_summary
from airweave.domains.taxonomy.tests.taxonomy_fixtures import publish
from airweave.domains.taxonomy.tests.test_artifacts import _fake_build

TENANT, READABLE = "t-acme", "neo-acme-1"
SOURCE, TARGET = "acme-2026.10", "acme-2026.11"


def write_active(s3: Path, version: str, sha: str, **fields) -> None:
    """The file as the platform writes it, with ``fields`` overriding or adding keys."""
    body = {
        "collection_readable_id": READABLE,
        "tenant_id": TENANT,
        "taxonomy_version": version,
        "version_prefix": f"taxonomy/tenants/{TENANT}/versions/{version}/",
        "manifest_sha256": sha,
        "generation": 3,
        "updated_at": "2026-10-15T00:00:00.000Z",
    }
    body.update(fields)
    body = {k: v for k, v in body.items() if v is not Ellipsis}
    path = s3 / "taxonomy/active"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{READABLE}.json").write_text(json.dumps(body))


@pytest.fixture
def world(tmp_path):
    s3 = tmp_path / "s3"
    built, cfg = _fake_build(tmp_path)
    source = publish(s3, built, cfg, TENANT, SOURCE)
    target = publish(s3, built, cfg, TENANT, TARGET)
    resolver = TaxonomyResolver(FilesystemBackend(s3), "taxonomy/", tmp_path / "local", 300)
    return s3, resolver, source, target


class TestLabellingVersion:
    async def test_migration_labels_with_the_target_and_ignores_serving(self, world):
        s3, resolver, _, target = world
        write_active(
            s3,
            TARGET,
            target.manifest_sha256,
            serving_version=SOURCE,
            labelling_version=TARGET,
            migration_id="mig-1",
        )
        t = await resolver.resolve(READABLE)
        assert t.taxonomy_version == TARGET and t.manifest_sha256 == target.manifest_sha256
        assert t.serving_version == SOURCE and t.migration_id == "mig-1"
        assert t.generation == 3

    async def test_the_labelling_version_decides_even_if_taxonomy_version_is_absent(self, world):
        s3, resolver, _, target = world
        write_active(
            s3,
            ...,
            target.manifest_sha256,
            labelling_version=TARGET,
            serving_version=SOURCE,
            version_prefix=f"taxonomy/tenants/{TENANT}/versions/{TARGET}/",
        )
        assert (await resolver.resolve(READABLE)).taxonomy_version == TARGET

    async def test_serving_version_alone_never_selects_the_classifier(self, world):
        s3, resolver, source, target = world
        # labelling is the target; the file's folder and checksum are the target's, not serving's
        write_active(
            s3, TARGET, target.manifest_sha256, serving_version=SOURCE, labelling_version=TARGET
        )
        t = await resolver.resolve(READABLE)
        assert t.cache_key == (TARGET, target.manifest_sha256)

    async def test_normal_state_both_equal(self, world):
        s3, resolver, source, _ = world
        write_active(
            s3,
            SOURCE,
            source.manifest_sha256,
            serving_version=SOURCE,
            labelling_version=SOURCE,
            migration_id=None,
        )
        t = await resolver.resolve(READABLE)
        assert (t.taxonomy_version, t.serving_version, t.migration_id) == (SOURCE, SOURCE, None)

    async def test_flip_state_serving_moves_labelling_stays(self, world):
        s3, resolver, _, target = world
        write_active(
            s3,
            TARGET,
            target.manifest_sha256,
            serving_version=TARGET,
            labelling_version=TARGET,
            migration_id="mig-1",
        )
        assert (await resolver.resolve(READABLE)).taxonomy_version == TARGET


class TestOlderFile:
    async def test_without_the_new_fields_labelling_is_taxonomy_version(self, world):
        s3, resolver, source, _ = world
        write_active(s3, SOURCE, source.manifest_sha256)
        t = await resolver.resolve(READABLE)
        assert t.taxonomy_version == SOURCE
        assert t.serving_version is None and t.migration_id is None

    async def test_only_labelling_version_given(self, world):
        s3, resolver, source, _ = world
        write_active(
            s3,
            ...,
            source.manifest_sha256,
            labelling_version=SOURCE,
            version_prefix=f"taxonomy/tenants/{TENANT}/versions/{SOURCE}/",
        )
        assert (await resolver.resolve(READABLE)).taxonomy_version == SOURCE


class TestRefused:
    async def test_taxonomy_version_and_labelling_version_disagree(self, world):
        s3, resolver, source, target = world
        write_active(s3, SOURCE, source.manifest_sha256, labelling_version=TARGET)
        with pytest.raises(TaxonomyResolutionError, match="labelling_version"):
            await resolver.resolve(READABLE)

    async def test_folder_must_be_the_labelling_versions(self, world):
        s3, resolver, source, target = world
        write_active(
            s3,
            TARGET,
            target.manifest_sha256,
            labelling_version=TARGET,
            serving_version=SOURCE,
            version_prefix=f"taxonomy/tenants/{TENANT}/versions/{SOURCE}/",
        )
        with pytest.raises(TaxonomyResolutionError, match="version_prefix"):
            await resolver.resolve(READABLE)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("labelling_version", "../x"),
            ("serving_version", "../x"),
            ("migration_id", "a/b"),
            ("migration_id", 7),
        ],
    )
    async def test_unsafe_values(self, world, field, value):
        s3, resolver, source, _ = world
        write_active(s3, SOURCE, source.manifest_sha256, **{field: value})
        with pytest.raises(TaxonomyResolutionError):
            await resolver.resolve(READABLE)


class TestLoggedWithEachSync:
    def test_summary_names_version_generation_and_migration(self, caplog):
        tally = TaxonomyTally()
        tally.record("job-1", {"ASSIGNED": 3})
        tally.note_version("job-1", TARGET, 3, "mig-1", SOURCE)
        logger = logging.getLogger("summary-test")
        with caplog.at_level("INFO", logger="summary-test"):
            log_sync_summary("job-1", logger, tally)
        text = caplog.records[-1].getMessage()
        assert f"{TARGET} (generation 3, migration mig-1, serving {SOURCE})" in text

    def test_summary_without_a_migration(self, caplog):
        tally = TaxonomyTally()
        tally.record("job-2", {"ASSIGNED": 3})
        tally.note_version("job-2", SOURCE, 1)
        logger = logging.getLogger("summary-test")
        with caplog.at_level("INFO", logger="summary-test"):
            log_sync_summary("job-2", logger, tally)
        text = caplog.records[-1].getMessage()
        assert f"{SOURCE} (generation 1)" in text and "migration" not in text

    def test_two_generations_in_one_sync_are_both_listed(self, caplog):
        tally = TaxonomyTally()
        tally.record("job-3", {"ASSIGNED": 3})
        tally.note_version("job-3", SOURCE, 1)
        tally.note_version("job-3", TARGET, 2, "mig-1", SOURCE)
        logger = logging.getLogger("summary-test")
        with caplog.at_level("INFO", logger="summary-test"):
            log_sync_summary("job-3", logger, tally)
        text = caplog.records[-1].getMessage()
        assert "generation 1" in text and "generation 2, migration mig-1" in text
