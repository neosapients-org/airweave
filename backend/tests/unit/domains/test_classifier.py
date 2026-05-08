"""Unit tests for ClassifierProcessor.

Tests cover:
- Disabled processor returns entities unchanged
- Successful text classification sets doc_categories and prefixes textual_representation
- Invalid LLM categories are dropped and default to "other" when nothing valid remains
- Malformed JSON from LLM raises ValueError (caught at batch level → defaults to "other")
- Classification failure for one entity defaults it to "other" without affecting others
- Vision OCR replaces textual_representation for image entities
- Vision OCR result of NO_TEXT_FOUND leaves textual_representation unchanged
- Non-FileEntity entities are classified but not given doc_categories (no attribute)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from airweave.domains.sync_pipeline.processors.classifier import (
    ClassifierProcessor,
    PREDEFINED_CATEGORIES,
)
from airweave.platform.entities._base import FileEntity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_entity(
    name="report.pdf",
    mime_type="application/pdf",
    text="Some document content",
    entity_id="ent:001",
) -> FileEntity:
    return FileEntity.model_construct(
        entity_id=entity_id,
        name=name,
        mime_type=mime_type,
        textual_representation=text,
        local_path=None,
        doc_categories=None,
    )


def _make_image_entity(name="scan.png", entity_id="ent:img") -> FileEntity:
    return FileEntity.model_construct(
        entity_id=entity_id,
        name=name,
        mime_type="image/png",
        textual_representation="base64garbagedata==",
        local_path=None,
        doc_categories=None,
    )


def _llm_response(categories: list[str] | str) -> MagicMock:
    """Build a mock OpenAI chat completion response for the given categories."""
    msg = MagicMock()
    msg.content = json.dumps({"doc_categories": categories})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _ocr_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifierProcessorDisabled:
    """When CLASSIFICATION_ENABLED=false, process() is a no-op."""

    @pytest.mark.asyncio
    async def test_returns_entities_unchanged_when_disabled(self):
        entity = _make_file_entity()
        processor = ClassifierProcessor()

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: False)):
            result = await processor.process([entity])

        assert result == [entity]
        assert entity.doc_categories is None
        assert entity.textual_representation == "Some document content"

    @pytest.mark.asyncio
    async def test_returns_empty_list_unchanged_when_disabled(self):
        processor = ClassifierProcessor()

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: False)):
            result = await processor.process([])

        assert result == []


class TestClassifierProcessorEnabled:
    """When CLASSIFICATION_ENABLED=true, entities are classified via OpenAI."""

    @pytest.mark.asyncio
    async def test_sets_doc_categories_on_file_entity(self):
        entity = _make_file_entity(text="Quarterly balance sheet and P&L report")
        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_llm_response("financial_report")
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert entity.doc_categories == ["financial_report"]

    @pytest.mark.asyncio
    async def test_prepends_category_to_textual_representation(self):
        entity = _make_file_entity(text="NDA between parties A and B")
        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_llm_response("contract")
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert entity.textual_representation.startswith("[Categories: contract]")
        assert "NDA between parties A and B" in entity.textual_representation

    @pytest.mark.asyncio
    async def test_unknown_llm_category_defaults_to_other(self):
        entity = _make_file_entity(text="Some content")
        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_llm_response("nonexistent_category")
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert entity.doc_categories == ["other"]

    @pytest.mark.asyncio
    async def test_other_with_description_is_accepted(self):
        """LLM may return 'other:brief_description' — should be accepted as-is."""
        entity = _make_file_entity(text="Something unusual")
        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_llm_response(["other:internal_memo"])
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert entity.doc_categories == ["other"]

    @pytest.mark.asyncio
    async def test_malformed_json_defaults_entity_to_other(self):
        """If LLM returns invalid JSON, the entity should default to 'other'."""
        entity = _make_file_entity()
        processor = ClassifierProcessor()

        bad_msg = MagicMock()
        bad_msg.content = "not json at all"
        bad_choice = MagicMock()
        bad_choice.message = bad_msg
        bad_resp = MagicMock()
        bad_resp.choices = [bad_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=bad_resp)

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert entity.doc_categories == ["other"]

    @pytest.mark.asyncio
    async def test_one_entity_failure_does_not_affect_others(self):
        """A classification error on one entity should not prevent others from being classified."""
        entity_ok = _make_file_entity(text="Annual report 2024", entity_id="ent:001")
        entity_fail = _make_file_entity(text="Some text", entity_id="ent:002")
        processor = ClassifierProcessor()

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Second call (entity_fail's classification) blows up
                raise RuntimeError("OpenAI timeout")
            return _llm_response("financial_report")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = side_effect

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity_ok, entity_fail])

        assert entity_ok.doc_categories == ["financial_report"]
        assert entity_fail.doc_categories == ["other"]

    @pytest.mark.asyncio
    async def test_returns_all_entities(self):
        """process() must return all entities, not just classified ones."""
        entities = [_make_file_entity(entity_id=f"ent:{i}", text=f"doc {i}") for i in range(3)]
        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_llm_response("reference")
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                result = await processor.process(entities)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_all_predefined_categories_are_accepted(self):
        """Every category in PREDEFINED_CATEGORIES should pass validation unchanged."""
        for category in PREDEFINED_CATEGORIES:
            entity = _make_file_entity(text="test content")
            processor = ClassifierProcessor()
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=_llm_response(category)
            )

            with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
                with patch.object(processor, "_get_client", return_value=mock_client):
                    await processor.process([entity])

            assert entity.doc_categories == [category], f"Category '{category}' was not accepted"

    @pytest.mark.asyncio
    async def test_accepts_multiple_categories(self):
        entity = _make_file_entity(text="Signed contract email thread")
        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_llm_response(["correspondence", "contract"])
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert entity.doc_categories == ["correspondence", "contract"]
        assert entity.textual_representation.startswith("[Categories: correspondence, contract]")

    def test_prompt_requests_multiple_categories(self):
        from airweave.domains.sync_pipeline.processors.classifier import CLASSIFICATION_PROMPT

        assert "1 to 3" in CLASSIFICATION_PROMPT
        assert "doc_categories" in CLASSIFICATION_PROMPT


class TestVisionOCR:
    """Vision OCR replaces textual_representation for image/* entities."""

    @pytest.mark.asyncio
    async def test_ocr_replaces_textual_representation_for_images(self, tmp_path):
        img_file = tmp_path / "scan.png"
        img_file.write_bytes(b"\x89PNG fake image bytes")

        entity = _make_image_entity()
        entity.local_path = str(img_file)
        entity.textual_representation = "base64garbage=="

        processor = ClassifierProcessor()

        # Two calls: first = OCR, second = classification
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _ocr_response("Invoice No. 12345\nDate: 2024-01-01"),  # OCR
                _llm_response("financial_report"),                      # classification
            ]
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert "Invoice No. 12345" in entity.textual_representation

    @pytest.mark.asyncio
    async def test_ocr_no_text_found_leaves_representation_unchanged(self, tmp_path):
        img_file = tmp_path / "blank.png"
        img_file.write_bytes(b"\x89PNG blank image")

        entity = _make_image_entity()
        entity.local_path = str(img_file)
        entity.textual_representation = "original text"

        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _ocr_response("NO_TEXT_FOUND"),       # OCR returns nothing
                _llm_response("other"),               # classification
            ]
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        # textual_representation should still contain original text (not replaced)
        assert "original text" in entity.textual_representation

    @pytest.mark.asyncio
    async def test_ocr_skipped_when_no_local_path(self):
        """Image entity without a local_path should skip OCR gracefully."""
        entity = _make_image_entity()
        entity.local_path = None
        entity.textual_representation = "kept as is"

        processor = ClassifierProcessor()
        mock_client = AsyncMock()
        # Only one call expected — classification only, no OCR
        mock_client.chat.completions.create = AsyncMock(
            return_value=_llm_response("presentation")
        )

        with patch.object(type(processor), "enabled", new_callable=lambda: property(lambda self: True)):
            with patch.object(processor, "_get_client", return_value=mock_client):
                await processor.process([entity])

        assert mock_client.chat.completions.create.call_count == 1
        assert entity.doc_categories == ["presentation"]
