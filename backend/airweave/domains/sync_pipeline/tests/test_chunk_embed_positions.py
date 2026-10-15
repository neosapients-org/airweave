"""ChunkEmbedProcessor chunk selection and positions (Airweave A3, contract section 3)."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from airweave.core.config import settings
from airweave.domains.converters.fakes.registry import FakeConverterRegistry
from airweave.domains.sync_pipeline.processors.chunk_embed import ChunkEmbedProcessor
from airweave.domains.sync_pipeline.processors.chunk_layout import ChunkLayout
from airweave.platform.destinations.vespa.transformer import CHUNK_LAYOUT_FIELDS
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb
from airweave.platform.entities.stub import PdfStubFileEntity

_SEMANTIC_CHUNKER = "airweave.platform.chunkers.semantic.SemanticChunker"

DOCUMENT = (
    "# Metadata\n\n**Source**: google_drive\n**Name**: Acme Q3.pdf\n\n# Content\n\n"
    "Acme Corp's late payments rose 12% in the quarter. The allowance for credit losses "
    "increased to €4.2 million. Globex paid its invoices on time. "
    "売上高は前年同期比で増加しました。"
    "\nInitech cut its backlog by a third. "
)


def _entity(text: str = DOCUMENT) -> PdfStubFileEntity:
    return PdfStubFileEntity(
        entity_id="acme-q3",
        stub_id="acme-q3",
        file_name="Acme Q3.pdf",
        breadcrumbs=[Breadcrumb(entity_id="f1", name="Finance", entity_type="Folder")],
        url="stub://acme-q3",
        size=10,
        file_type="pdf",
        textual_representation=text,
        airweave_system_metadata=AirweaveSystemMetadata(
            source_name="google_drive",
            sync_id=UUID("5c0b9a6e-0000-4000-8000-000000000001"),
            hash="h-1",
        ),
    )


@pytest.fixture
def processor():
    dense, sparse = MagicMock(), MagicMock()
    dense.dimensions = 3
    return ChunkEmbedProcessor(
        converter_registry=FakeConverterRegistry(), dense_embedder=dense, sparse_embedder=sparse
    )


@pytest.fixture
def sync_context():
    ctx = MagicMock()
    ctx.source_short_name = "google_drive"
    return ctx


@pytest.fixture
def sentence_pack(monkeypatch):
    monkeypatch.setattr(settings, "CHUNKER_MODE", "sentence_pack")
    monkeypatch.setattr(settings, "CHUNK_TARGET_TOKENS", 40)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP_TOKENS", 16)


class TestDefaultMode:
    """With the upstream settings the semantic chunker's output passes through unchanged."""

    async def test_semantic_texts_unchanged_and_no_new_fields(self, processor, sync_context):
        chunks = [{"text": "Chunk zero."}, {"text": "   "}, {"text": "Chunk two."}]
        with patch(_SEMANTIC_CHUNKER) as cls:
            cls.return_value.chunk_batch = AsyncMock(return_value=[chunks])
            out = await processor._chunk_textual_entities([_entity()], sync_context)

        # whitespace-only chunk dropped and index gap kept, exactly as upstream
        assert [c.textual_representation for c in out] == ["Chunk zero.", "Chunk two."]
        assert [c.airweave_system_metadata.chunk_index for c in out] == [0, 2]
        assert [c.entity_id for c in out] == ["acme-q3__chunk_0", "acme-q3__chunk_2"]
        for c in out:
            assert all(getattr(c, name) is None for name in CHUNK_LAYOUT_FIELDS)

    async def test_default_mode_never_uses_sentence_pack(self, processor, sync_context):
        with (
            patch(_SEMANTIC_CHUNKER) as cls,
            patch(
                "airweave.domains.sync_pipeline.processors.chunk_embed.SentencePackChunker"
            ) as pack_cls,
        ):
            cls.return_value.chunk_batch = AsyncMock(return_value=[[{"text": "x"}]])
            await processor._chunk_textual_entities([_entity()], sync_context)
        pack_cls.assert_not_called()


class TestSentencePackMode:
    async def test_positions_are_exact_slices(self, processor, sync_context, sentence_pack):
        entity = _entity()
        out = await processor._chunk_textual_entities([entity], sync_context)
        assert len(out) == 3
        for i, c in enumerate(out):
            assert c.textual_representation == DOCUMENT[c.chunk_char_start : c.chunk_char_end]
            assert c.chunk_own_start == (out[i - 1].chunk_char_end if i else c.chunk_char_start)
            assert c.chunk_body_offset == 0  # no header
            assert c.airweave_system_metadata.chunk_index == i
            assert c.entity_id == f"acme-q3__chunk_{i}"
            assert c.airweave_system_metadata.original_entity_id == "acme-q3"
        assert out[0].chunk_char_start == 0 and out[-1].chunk_char_end == len(DOCUMENT)
        assert "".join(DOCUMENT[c.chunk_own_start : c.chunk_char_end] for c in out) == DOCUMENT

    async def test_overlap_present(self, processor, sync_context, sentence_pack):
        out = await processor._chunk_textual_entities([_entity()], sync_context)
        assert any(c.chunk_own_start > c.chunk_char_start for c in out)

    async def test_parent_entity_untouched(self, processor, sync_context, sentence_pack):
        entity = _entity()
        await processor._chunk_textual_entities([entity], sync_context)
        assert all(getattr(entity, name) is None for name in CHUNK_LAYOUT_FIELDS)

    def test_whitespace_chunk_kept_so_own_ranges_tile(self, processor, sync_context):
        """With positions, a whitespace-only chunk is kept rather than leaving a gap."""
        chunks = [
            {"text": "Acme.", "start_index": 0, "end_index": 5, "own_start": 0},
            {"text": "   ", "start_index": 5, "end_index": 8, "own_start": 5},
        ]
        out = processor._multiply_entities(
            [_entity("Acme.   ")], [chunks], sync_context, ChunkLayout(positions=True)
        )
        assert [(c.chunk_char_start, c.chunk_char_end) for c in out] == [(0, 5), (5, 8)]
