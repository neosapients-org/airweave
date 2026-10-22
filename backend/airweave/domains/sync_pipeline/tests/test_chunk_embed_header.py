"""The header line on every chunk and the keyword input (Airweave A5, contract 3.1 and 3.2)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from airweave.core.config import chunking, settings
from airweave.domains.converters.fakes.registry import FakeConverterRegistry
from airweave.domains.converters.page_markers import page_marker
from airweave.domains.sync_pipeline.processors.chunk_embed import ChunkEmbedProcessor
from airweave.domains.sync_pipeline.processors.chunk_layout import build_header
from airweave.platform.destinations.vespa.transformer import CHUNK_LAYOUT_FIELDS, EntityTransformer
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb
from airweave.platform.entities.stub import PdfStubFileEntity

_SEMANTIC_CHUNKER = "airweave.platform.chunkers.semantic.SemanticChunker"
TEXT = "Acme Corp's late payments rose 12% in the quarter. " * 6 + "Globex paid on time. " * 6


def _entity(text: str, name: str = "Q3 credit risk report.pdf", path=("Finance", "Risk", "2026")):
    return PdfStubFileEntity(
        entity_id="f3a1",
        name=name,  # set from the flagged field by the entity pipeline in a sync
        stub_id="f3a1",
        file_name=name,
        breadcrumbs=[Breadcrumb(entity_id=p, name=p, entity_type="Folder") for p in path],
        url="stub://x",
        size=1,
        file_type="pdf",
        textual_representation=text,
        airweave_system_metadata=AirweaveSystemMetadata(
            source_name="google_drive", sync_id=UUID("5c0b9a6e-0000-4000-8000-000000000001")
        ),
    )


@pytest.fixture
def processor():
    dense, sparse = MagicMock(), MagicMock()
    dense.dimensions = 3
    dense.embed_many = AsyncMock(
        side_effect=lambda texts: [SimpleNamespace(vector=[0.1, 0.2, 0.3]) for _ in texts]
    )
    sparse.embed_many = AsyncMock(side_effect=lambda texts: [MagicMock() for _ in texts])
    return ChunkEmbedProcessor(FakeConverterRegistry(), dense, sparse)


@pytest.fixture
def sync_context():
    ctx = MagicMock()
    ctx.source_short_name = "google_drive"
    return ctx


@pytest.fixture
def neo(monkeypatch):
    """The Neo chunking at test size: a header follows the mode, the overlap is a share of the target."""

    def _set(mode: str = "sentence_pack", target: int = 40, overlap: int = 16):
        monkeypatch.setattr(settings, "CHUNKER_MODE", mode)
        monkeypatch.setattr(settings, "CHUNK_TARGET_TOKENS", target)
        monkeypatch.setattr(chunking, "OVERLAP_PERCENT", overlap * 100 // target)

    return _set


class TestBuildHeader:
    def test_format(self):
        crumbs = [Breadcrumb(entity_id=p, name=p, entity_type="F") for p in ("Finance", "Risk")]
        expected = "a.pdf · google_drive · Finance / Risk"
        assert build_header("a.pdf", "google_drive", crumbs) == expected
        assert build_header("a.pdf", "s", crumbs, 12, 12) == "a.pdf · s · Finance / Risk · page 12"
        assert build_header("a.pdf", "s", crumbs, 12, 13).endswith(" · pages 12–13")

    def test_no_path(self):
        assert build_header("a.pdf", "slack", []) == "a.pdf · slack"

    def test_one_line_and_clean(self):
        header = build_header("Q3\nreport\r\n .pdf\x03", "s", [])
        assert "\n" not in header and "\r" not in header and "\x03" not in header
        assert header.splitlines() == [header]
        assert len(header) == len("Q3\nreport\r\n .pdf\x03 · s")  # one for one


class TestSharedFixture:
    async def test_textual_representation_and_body_offset_match(
        self, processor, sync_context, neo, airweave_neo_chunk_fixture
    ):
        fx = airweave_neo_chunk_fixture
        neo(
            "sentence_pack",
            fx["settings"]["CHUNK_TARGET_TOKENS"],
            fx["settings"]["CHUNK_OVERLAP_TOKENS"],
        )
        text = fx["source_text"]
        for start in sorted(fx["page_starts"], key=lambda s: s["offset"], reverse=True):
            text = text[: start["offset"]] + page_marker(start["page"]) + text[start["offset"] :]
        out = await processor._chunk_textual_entities([_entity(text)], sync_context)
        want = fx["chunks"]
        got = [c.textual_representation for c in out]
        assert got == [c["textual_representation"] for c in want]
        assert [c.chunk_body_offset for c in out] == [c["chunk_body_offset"] for c in want]
        for c, w in zip(out, want, strict=True):
            assert c.textual_representation[c.chunk_body_offset :] == w["_test"]["body"]


class TestHeaderSwitch:
    async def test_enabled_every_chunk_starts_with_header(self, processor, sync_context, neo):
        neo()
        out = await processor._chunk_textual_entities([_entity(TEXT)], sync_context)
        header = "Q3 credit risk report.pdf · google_drive · Finance / Risk / 2026"
        assert len(out) > 2
        for c in out:
            assert c.textual_representation.startswith(header + "\n")
            assert (
                c.textual_representation[c.chunk_body_offset :]
                == TEXT[c.chunk_char_start : c.chunk_char_end]
            )

    async def test_name_with_newline_and_refused_character(self, processor, sync_context, neo):
        neo()
        entity = _entity(TEXT, name="Q3\ncredit risk\x03.pdf")
        out = await processor._chunk_textual_entities([entity], sync_context)
        for c in out:
            first_line = c.textual_representation.split("\n", 1)[0]
            assert len(first_line) + 1 == c.chunk_body_offset
            assert (
                c.textual_representation[c.chunk_body_offset :]
                == TEXT[c.chunk_char_start : c.chunk_char_end]
            )
            # the transformer's own cleaning leaves the header alone, so the offset holds
            doc = EntityTransformer().transform(c)
            stored = doc.fields["textual_representation"]
            assert stored[c.chunk_body_offset :] == TEXT[c.chunk_char_start : c.chunk_char_end]

    async def test_upstream_settings_no_header_no_fields(self, processor, sync_context):
        with patch(_SEMANTIC_CHUNKER) as cls:
            cls.return_value.chunk_batch = AsyncMock(return_value=[[{"text": "Body text."}]])
            (c,) = await processor._chunk_textual_entities([_entity(TEXT)], sync_context)
        assert c.textual_representation == "Body text."
        assert all(getattr(c, name) is None for name in CHUNK_LAYOUT_FIELDS)

    async def test_pages_in_header(self, processor, sync_context, neo):
        neo()
        text = page_marker(3) + TEXT[:300] + page_marker(4) + TEXT[300:]
        out = await processor._chunk_textual_entities([_entity(text)], sync_context)
        headers = {c.textual_representation.split("\n", 1)[0].rsplit(" · ", 1)[1] for c in out}
        assert headers <= {"page 3", "page 4", "pages 3–4"} and "pages 3–4" in headers


class TestEmbeddingInputs:
    async def test_dense_input_has_header_sparse_input_has_no_new_fields(
        self, processor, sync_context, neo
    ):
        neo()
        chunks = await processor._chunk_textual_entities([_entity(TEXT)], sync_context)
        for c in chunks:
            c.page_from, c.topic_ids, c.taxonomy_status = 1, {"acme:credit.x": 1}, "ASSIGNED"
        await processor._embed_entities(chunks, sync_context)

        dense_texts = processor._dense_embedder.embed_many.call_args.args[0]
        assert all(t.startswith("Q3 credit risk report.pdf · ") for t in dense_texts)
        sparse_texts = processor._sparse_embedder.embed_many.call_args.args[0]
        for raw in sparse_texts:
            assert not set(CHUNK_LAYOUT_FIELDS) & set(json.loads(raw))

    async def test_code_files_never_get_a_header(self, processor, sync_context, neo):
        neo()
        entity = _entity("def f():\n    return 1\n")
        out = processor._multiply_entities(
            [entity], [[{"text": "def f():\n    return 1\n"}]], sync_context
        )
        assert out[0].textual_representation == "def f():\n    return 1\n"
        assert out[0].chunk_body_offset is None


class TestHeaderTokenBudget:
    """The header is embedded with the body: it has a token budget of its own."""

    @staticmethod
    def count(text: str) -> int:
        # one token per character is the worst case a tokenizer can reach (CJK, symbols)
        return len(text)

    def test_a_header_within_the_budget_is_unchanged(self):
        from airweave.domains.sync_pipeline.processors.chunk_layout import fit_header

        assert fit_header("report.pdf · drive", 50, self.count) == "report.pdf · drive"

    def test_a_longer_header_is_cut_to_the_budget_and_ends_in_an_ellipsis(self):
        from airweave.domains.sync_pipeline.processors.chunk_layout import fit_header

        header = "x" * 500 + " · google_drive"

        fitted = fit_header(header, 40, self.count)

        assert self.count(fitted) <= 40 and fitted.endswith("…") and fitted.startswith("xxx")
        assert "\n" not in fitted

    def test_the_real_tokenizer_bounds_dense_text_that_a_character_limit_would_not(self):
        from airweave.domains.sync_pipeline.processors.chunk_layout import (
            _header_tokenizer,
            fit_header,
        )

        header = "契約書の名前" * 200  # CJK: more than one token per character
        count = _header_tokenizer().count_tokens

        fitted = fit_header(header, 64)

        assert count(header) > 64 and count(fitted) <= 64

    async def test_a_chunk_never_carries_a_header_over_the_budget(
        self, processor, sync_context, neo, monkeypatch
    ):
        from airweave.domains.sync_pipeline.processors import chunk_layout
        from airweave.domains.sync_pipeline.processors.chunk_layout import _header_tokenizer

        neo()
        monkeypatch.setattr(chunk_layout, "HEADER_MAX_TOKENS", 32)
        entity = _entity(TEXT, name="n" * 400)

        chunks = await processor._chunk_textual_entities([entity], sync_context)

        assert chunks
        count = _header_tokenizer().count_tokens
        for chunk in chunks:
            header = chunk.textual_representation[: chunk.chunk_body_offset - 1]
            assert count(header) <= 32 and header.endswith("…") is False or count(header) <= 32
            assert len(header) < 400
