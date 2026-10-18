"""The chunk fingerprint in the entity hash (Airweave A6, contract section 9)."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from airweave.core.config import chunking, settings
from airweave.domains.sync_pipeline.entity.actions import EntityKeepAction, EntityUpdateAction
from airweave.domains.sync_pipeline.entity.resolver import EntityActionResolver
from airweave.domains.sync_pipeline.pipeline.hash_computer import HashComputer, chunk_fingerprint
from airweave.platform.entities._airweave_field import AirweaveField
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb, CodeFileEntity
from airweave.platform.entities.stub import PdfStubFileEntity, SmallStubEntity

# Hashes of the two entities below computed by upstream's HashComputer (370c3e605): with every
# chunk setting at its default, the fork must hash exactly as upstream did.
UPSTREAM_PLAIN_HASH = "c539757d9e79245ae691fb4dc2f7b3e866911f55217b1c5dec4dd82a95eec522"
UPSTREAM_PDF_HASH = "afd500577b15b868049abf3d061538ee44c1c942407163ea3cd23a14fe7d68b5"

# The two settings a deployment makes; the overlap, the header and the version follow from them.
NEO = {"CHUNKER_MODE": "sentence_pack", "CHUNK_TARGET_TOKENS": 300}


def _plain() -> SmallStubEntity:
    return SmallStubEntity(
        stub_id="s1", title="Acme note", content="Globex invoice paid.", breadcrumbs=[]
    )


def _pdf(tmp_path) -> PdfStubFileEntity:
    path = tmp_path / "initech.pdf"
    path.write_bytes(b"%PDF-1.4 Initech quarterly report body")
    return PdfStubFileEntity(
        stub_id="p1",
        file_name="Initech Q3.pdf",
        url="stub://p1",
        size=38,
        file_type="pdf",
        local_path=str(path),
        breadcrumbs=[Breadcrumb(entity_id="f", name="Finance", entity_type="Folder")],
    )


class _CodeEntity(CodeFileEntity):
    """Minimal code file entity."""

    code_id: str = AirweaveField(..., is_entity_id=True)
    code_name: str = AirweaveField(..., is_name=True)


def _code(tmp_path) -> CodeFileEntity:
    path = tmp_path / "acme.py"
    path.write_text("def acme():\n    return 1\n")
    return _CodeEntity(
        code_id="c1",
        code_name="acme.py",
        url="stub://c1",
        size=1,
        file_type="py",
        local_path=str(path),
        repo_name="acme",
        path_in_repo="acme.py",
        repo_owner="acme",
        language="python",
        commit_id="abc",
        breadcrumbs=[],
    )


@pytest.fixture
def chunk_settings(monkeypatch):
    def _set(**values):
        for name, value in values.items():
            monkeypatch.setattr(settings, name, value)

    return _set


class TestDefaults:
    async def test_hashes_equal_upstream(self, tmp_path):
        computer = HashComputer()
        assert await computer.compute_for_entity(_plain()) == UPSTREAM_PLAIN_HASH
        assert await computer.compute_for_entity(_pdf(tmp_path)) == UPSTREAM_PDF_HASH

    def test_no_fingerprint_at_defaults(self):
        assert chunk_fingerprint(_plain()) is None


class TestNeoSettings:
    async def test_every_chunk_setting_changes_the_hash(self, tmp_path, chunk_settings):
        computer = HashComputer()
        base = await computer.compute_for_entity(_plain())
        seen = {base}
        for name, value in NEO.items():
            chunk_settings(**{name: value})
            h = await computer.compute_for_entity(_plain())
            assert h not in seen, f"{name} did not change the hash"
            seen.add(h)

    async def test_stable_under_the_same_settings(self, tmp_path, chunk_settings):
        chunk_settings(**NEO)
        computer = HashComputer()
        assert await computer.compute_for_entity(
            _pdf(tmp_path)
        ) == await computer.compute_for_entity(_pdf(tmp_path))

    async def test_only_target_changed(self, tmp_path, chunk_settings):
        """300 -> 600 alone re-chunks every file once."""
        computer = HashComputer()
        chunk_settings(**NEO)
        at_300 = await computer.compute_for_entity(_pdf(tmp_path))
        chunk_settings(CHUNK_TARGET_TOKENS=600)
        assert await computer.compute_for_entity(_pdf(tmp_path)) != at_300

    async def test_the_chunker_version_is_part_of_the_hash(
        self, tmp_path, monkeypatch, chunk_settings
    ):
        """A new chunker version re-chunks every text entity once, whatever the settings."""
        computer = HashComputer()
        chunk_settings(**NEO)
        before = await computer.compute_for_entity(_pdf(tmp_path))
        monkeypatch.setattr(chunking, "SENTENCE_PACK_VERSION", "3")
        assert await computer.compute_for_entity(_pdf(tmp_path)) != before

    async def test_code_files_never_fingerprinted(self, tmp_path, chunk_settings):
        computer = HashComputer()
        before = await computer.compute_for_entity(_code(tmp_path))
        chunk_settings(**NEO)
        assert chunk_fingerprint(_code(tmp_path)) is None
        assert await computer.compute_for_entity(_code(tmp_path)) == before


class TestResync:
    """Switching the Neo settings on updates every entity once; the next sync keeps them."""

    @staticmethod
    def _resolve(resolver, entity, db_hash):
        db_row = MagicMock()
        db_row.hash, db_row.id = db_hash, uuid4()
        existing = {(entity.entity_id, "stub"): db_row}
        return resolver._resolve_non_delete_action(entity, existing, MagicMock())

    async def test_update_once_then_keep(self, tmp_path, chunk_settings):
        from airweave.domains.entities.registry import EntityDefinitionRegistry

        registry = EntityDefinitionRegistry()
        registry._by_class = {PdfStubFileEntity: "stub"}
        resolver = EntityActionResolver(entity_registry=registry, entity_repo=MagicMock())
        computer = HashComputer()

        def entity(h):
            e = _pdf(tmp_path)
            e.entity_id = "p1"
            e.airweave_system_metadata = AirweaveSystemMetadata(hash=h)
            return e

        stored = await computer.compute_for_entity(_pdf(tmp_path))  # last sync, defaults
        chunk_settings(**NEO)
        first = await computer.compute_for_entity(_pdf(tmp_path))
        assert isinstance(self._resolve(resolver, entity(first), stored), EntityUpdateAction)
        second = await computer.compute_for_entity(_pdf(tmp_path))
        assert isinstance(self._resolve(resolver, entity(second), first), EntityKeepAction)
