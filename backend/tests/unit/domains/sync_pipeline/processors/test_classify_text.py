"""Unit tests for standalone classify_text helper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_classify_text_returns_list():
    """classify_text must return a list of categories."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"doc_categories": ["correspondence"]}'

    with patch(
        "airweave.domains.sync_pipeline.processors.classifier.settings.OPENAI_API_KEY",
        "test-key",
    ):
        with patch(
            "airweave.domains.sync_pipeline.processors.classifier.AsyncOpenAI"
        ) as mock_openai:
            mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            from airweave.domains.sync_pipeline.processors.classifier import classify_text

            result = await classify_text(
                text="Hello, please find the attached invoice.",
                filename="invoice.pdf",
                mime_type="application/pdf",
            )

    assert isinstance(result, list)
    assert result == ["correspondence"]
