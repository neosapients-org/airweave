"""Page markers, pages and refused characters in ChunkEmbedProcessor (Airweave A4).

Contract airweave-neo/1, sections 3.1, 3.2 and C19.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from airweave.core.config import chunking, settings
from airweave.domains.converters.fakes.registry import FakeConverterRegistry
from airweave.domains.converters.page_markers import PAGE_SENTINEL, page_marker
from airweave.domains.sync_pipeline.processors.chunk_embed import ChunkEmbedProcessor
from airweave.domains.sync_pipeline.processors.chunk_layout import replace_refused_characters
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb
from airweave.platform.entities.stub import PdfStubFileEntity

_SEMANTIC_CHUNKER = "airweave.platform.chunkers.semantic.SemanticChunker"

# One code point of every class Vespa refuses (contract 3.2), and characters it keeps.
REFUSED = [
    "\x00",
    "\x03",
    "\x0b",
    "\x0c",
    "\x1f",
    "\ud800",
    "\udfff",
    "\ufdd0",
    "\ufdef",
    "\ufffe",
    "\uffff",
    "\U0001fffe",
    "\U0010ffff",
]
KEPT = ["\t", "\n", "\r", "\x7f", "\x80", "\x9f", "📈", "\ue000", "€", "売"]


def _entity(text: str, entity_id: str = "acme-q3") -> PdfStubFileEntity:
    return PdfStubFileEntity(
        entity_id=entity_id,
        stub_id=entity_id,
        file_name="Q3 credit risk report.pdf",
        breadcrumbs=[Breadcrumb(entity_id="d", name="Risk", entity_type="Folder")],
        url="stub://x",
        size=1,
        file_type="pdf",
        textual_representation=text,
        airweave_system_metadata=AirweaveSystemMetadata(
            source_name="google_drive", sync_id=UUID("5c0b9a6e-0000-4000-8000-000000000001")
        ),
    )


def _with_markers(text: str, page_starts: list) -> str:
    """Insert page markers at the given offsets, as a converter would have emitted them."""
    for start in sorted(page_starts, key=lambda s: s["offset"], reverse=True):
        text = text[: start["offset"]] + page_marker(start["page"]) + text[start["offset"] :]
    return text


@pytest.fixture
def processor():
    dense, sparse = MagicMock(), MagicMock()
    return ChunkEmbedProcessor(FakeConverterRegistry(), dense, sparse)


@pytest.fixture
def sync_context():
    ctx = MagicMock()
    ctx.source_short_name = "google_drive"
    return ctx


@pytest.fixture
def sentence_pack(monkeypatch):
    def _set(target: int = 40, overlap: int = 16):
        monkeypatch.setattr(settings, "CHUNKER_MODE", "sentence_pack")
        monkeypatch.setattr(settings, "CHUNK_TARGET_TOKENS", target)
        monkeypatch.setattr(chunking, "OVERLAP_PERCENT", overlap * 100 // target)

    return _set


class TestRefusedCharacters:
    def test_every_class_replaced_by_one_space(self):
        for ch in REFUSED:
            assert replace_refused_characters(f"a{ch}b") == "a b", repr(ch)

    def test_kept_characters_untouched(self):
        text = "".join(KEPT)
        assert replace_refused_characters(text) == text

    def test_fixture_source_text_cleans_to_document_text(self, airweave_neo_chunk_fixture):
        fx = airweave_neo_chunk_fixture
        assert replace_refused_characters(fx["source_text"]) == fx["document_text"]
        assert len(fx["source_text"]) == len(fx["document_text"])


class TestSharedFixture:
    """The fixture's source text, with its page markers, through the processor."""

    @pytest.fixture
    async def chunks(self, processor, sync_context, sentence_pack, airweave_neo_chunk_fixture):
        fx = airweave_neo_chunk_fixture
        sentence_pack(fx["settings"]["CHUNK_TARGET_TOKENS"], fx["settings"]["CHUNK_OVERLAP_TOKENS"])
        self.entity = _entity(_with_markers(fx["source_text"], fx["page_starts"]), "f3a1")
        return await processor._chunk_textual_entities([self.entity], sync_context)

    async def test_document_text_is_the_fixtures(self, chunks, airweave_neo_chunk_fixture):
        assert self.entity.textual_representation == airweave_neo_chunk_fixture["document_text"]

    async def test_ranges_pages_and_bodies(self, chunks, airweave_neo_chunk_fixture):
        want = airweave_neo_chunk_fixture["chunks"]
        got = [
            (c.chunk_char_start, c.chunk_char_end, c.chunk_own_start, c.page_from, c.page_to)
            for c in chunks
        ]
        assert got == [
            (
                c["chunk_char_start"],
                c["chunk_char_end"],
                c["chunk_own_start"],
                c["page_from"],
                c["page_to"],
            )
            for c in want
        ]
        bodies = [c.textual_representation[c.chunk_body_offset :] for c in chunks]
        assert bodies == [c["_test"]["body"] for c in want]
        assert [c.entity_id for c in chunks] == [c["entity_id"] for c in want]

    async def test_no_sentinel_or_refused_character_in_chunks(self, chunks):
        for c in chunks:
            assert PAGE_SENTINEL not in c.textual_representation
            assert replace_refused_characters(c.textual_representation) == c.textual_representation


class TestDefaultMode:
    async def test_markers_removed_and_nothing_else_changes(self, processor, sync_context):
        """Upstream mode: the semantic chunker sees upstream's text.

        Markers are gone, there are no pages, and refused characters are left for the
        transformer to remove, as before.
        """
        text = "Intro\x03 line\n" + page_marker(1) + "Page one.\n\n" + page_marker(2) + "Two."
        entity = _entity(text)
        with patch(_SEMANTIC_CHUNKER) as cls:
            cls.return_value.chunk_batch = AsyncMock(return_value=[[{"text": "Intro"}]])
            out = await processor._chunk_textual_entities([entity], sync_context)
            seen = cls.return_value.chunk_batch.call_args.args[0]
        assert seen == ["Intro\x03 line\nPage one.\n\nTwo."]
        assert out[0].page_from is None and out[0].page_to is None


class TestPagesOnChunks:
    async def test_format_without_pages_gets_none(self, processor, sync_context, sentence_pack):
        sentence_pack(40, 16)
        entity = _entity("Globex memo. " * 40)
        out = await processor._chunk_textual_entities([entity], sync_context)
        assert out and all(c.page_from is None and c.page_to is None for c in out)

    async def test_real_pdf_pages_match_their_text(self, tmp_path, sync_context, sentence_pack):
        """A generated PDF through the real converter and text builder.

        Every chunk's pages are the pages its text came from.
        """
        import fitz

        from airweave.domains.converters.registry import ConverterRegistry

        pages = [
            f"Acme page {n} sentence {i} about revenue and margins."
            for n in range(1, 4)
            for i in range(12)
        ]
        doc = fitz.open()
        for n in range(3):
            page = doc.new_page()
            page.insert_textbox(
                fitz.Rect(50, 50, 560, 800), " ".join(pages[n * 12 : n * 12 + 12]), fontsize=10
            )
        path = tmp_path / "acme.pdf"
        doc.save(path)
        doc.close()

        sentence_pack(40, 16)
        processor = ChunkEmbedProcessor(ConverterRegistry(), MagicMock(), MagicMock())
        entity = _entity("")
        entity.textual_representation = None
        entity.local_path = str(path)
        runtime = MagicMock()
        runtime.entity_tracker = AsyncMock()
        built = await processor._text_builder.build_for_batch([entity], sync_context, runtime)
        out = await processor._chunk_textual_entities(built, sync_context)

        text = built[0].textual_representation
        page2, page3 = text.index("Acme page 2"), text.index("Acme page 3")

        def page_of(i: int) -> int:
            return 1 if i < page2 else 2 if i < page3 else 3

        assert {c.page_from for c in out} == {1, 2, 3}
        for c in out:
            assert PAGE_SENTINEL not in c.textual_representation
            assert (c.page_from, c.page_to) == (
                page_of(c.chunk_char_start),
                page_of(c.chunk_char_end - 1),
            )
