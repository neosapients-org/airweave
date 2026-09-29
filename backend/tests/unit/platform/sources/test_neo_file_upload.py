"""Unit tests for NeoFileUploadSource."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from airweave.core.logging import ContextualLogger
from airweave.domains.sources.token_providers.protocol import SourceAuthProvider
from airweave.domains.storage.exceptions import FileSkippedException, StorageException
from airweave.platform.entities.neo_file_upload import NeoUploadedFileEntity
from airweave.platform.http_client.airweave_client import AirweaveHttpClient
from airweave.platform.sources.neo_file_upload import NeoFileUploadSource


@pytest.fixture
def fake_storage():
    """A minimal async storage backend double."""
    storage = MagicMock()
    storage.list_files = AsyncMock(return_value=[])
    storage.read_file = AsyncMock(return_value=b"")
    storage.exists = AsyncMock(return_value=True)
    return storage


@pytest.fixture
def source(fake_storage):
    """A NeoFileUploadSource wired to the fake storage backend."""
    s = NeoFileUploadSource(
        auth=MagicMock(spec=SourceAuthProvider),
        logger=MagicMock(spec=ContextualLogger),
        http_client=MagicMock(spec=AirweaveHttpClient),
    )
    s.upload_prefix = "uploads/org-1/collection-1"
    s._storage = fake_storage
    return s


@pytest.fixture
def files_service():
    """A FileService double whose save_bytes sets local_path like the real one."""
    files = MagicMock()

    async def _save_bytes(entity, content, filename_with_extension, logger):
        entity.local_path = f"/tmp/{filename_with_extension}"
        return entity

    files.save_bytes = AsyncMock(side_effect=_save_bytes)
    return files


class TestGenerateEntities:
    """generate_entities lists the upload prefix and yields one entity per file."""

    @pytest.mark.asyncio
    async def test_yields_entity_per_stored_file(self, source, fake_storage, files_service):
        fake_storage.list_files.return_value = [
            "uploads/org-1/collection-1/notes.md",
            "uploads/org-1/collection-1/folder/report.pdf",
        ]
        fake_storage.read_file.return_value = b"hello"

        entities = [e async for e in source.generate_entities(files=files_service)]

        assert len(entities) == 2
        by_id = {e.id: e for e in entities}
        assert by_id["notes.md"].name == "notes.md"
        assert by_id["notes.md"].folder_segments == []
        assert by_id["folder/report.pdf"].name == "report.pdf"
        assert by_id["folder/report.pdf"].folder_segments == ["folder"]
        assert all(isinstance(e, NeoUploadedFileEntity) for e in entities)

    @pytest.mark.asyncio
    async def test_uploaded_at_is_a_real_datetime_not_a_string(
        self, source, fake_storage, files_service
    ):
        """Regression test: `entity.created_at = <this value>` is a plain attribute
        assignment with no Pydantic coercion (see pipeline.py's flagged-field
        copy), and the Vespa transformer calls `.timestamp()` on it. A string
        here fails that call for every chunk, silently, while the sync still
        reports success — see the field's docstring in the entity module.
        """
        fake_storage.list_files.return_value = ["uploads/org-1/collection-1/notes.md"]
        fake_storage.read_file.return_value = b"hello"

        entities = [e async for e in source.generate_entities(files=files_service)]

        assert isinstance(entities[0].uploaded_at, datetime)

    @pytest.mark.asyncio
    async def test_strips_upload_prefix_from_entity_id(self, source, fake_storage, files_service):
        fake_storage.list_files.return_value = ["uploads/org-1/collection-1/a.txt"]
        fake_storage.read_file.return_value = b"x"

        entities = [e async for e in source.generate_entities(files=files_service)]

        assert entities[0].id == "a.txt"
        assert entities[0].relative_path == "a.txt"

    @pytest.mark.asyncio
    async def test_no_files_yields_nothing(self, source, fake_storage, files_service):
        fake_storage.list_files.return_value = []

        entities = [e async for e in source.generate_entities(files=files_service)]

        assert entities == []

    @pytest.mark.asyncio
    async def test_unreadable_file_is_skipped_not_raised(
        self, source, fake_storage, files_service
    ):
        fake_storage.list_files.return_value = [
            "uploads/org-1/collection-1/broken.md",
            "uploads/org-1/collection-1/ok.md",
        ]

        async def _read_file(path):
            if "broken" in path:
                raise StorageException("boom")
            return b"content"

        fake_storage.read_file.side_effect = _read_file

        entities = [e async for e in source.generate_entities(files=files_service)]

        assert len(entities) == 1
        assert entities[0].id == "ok.md"

    @pytest.mark.asyncio
    async def test_unsupported_extension_is_skipped_not_raised(
        self, source, fake_storage, files_service
    ):
        fake_storage.list_files.return_value = ["uploads/org-1/collection-1/archive.zip"]
        fake_storage.read_file.return_value = b"x"
        files_service.save_bytes = AsyncMock(
            side_effect=FileSkippedException(reason="Unsupported file extension", filename="archive.zip")
        )

        entities = [e async for e in source.generate_entities(files=files_service)]

        assert entities == []

    @pytest.mark.asyncio
    async def test_requires_file_service(self, source):
        with pytest.raises(AssertionError):
            async for _ in source.generate_entities(files=None):
                pass


class TestValidate:
    """validate() only needs the storage backend, no third-party call."""

    @pytest.mark.asyncio
    async def test_validate_checks_storage_reachable(self, source, fake_storage):
        await source.validate()
        fake_storage.exists.assert_awaited_once_with(source.upload_prefix)
