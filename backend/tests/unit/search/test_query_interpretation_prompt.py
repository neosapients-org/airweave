"""Unit tests for query interpretation temporal prompt guidance."""

from airweave.search.prompts import QUERY_INTERPRETATION_SYSTEM_PROMPT


def test_prompt_mentions_temporal_weight():
    """Prompt must instruct the LLM to set temporal_weight for recency queries."""
    prompt_lower = QUERY_INTERPRETATION_SYSTEM_PROMPT.lower()
    assert "temporal_weight" in prompt_lower
    assert "last" in prompt_lower or "recent" in prompt_lower


def test_prompt_gives_temporal_weight_examples():
    """Prompt must include concrete recency examples."""
    has_example = any(
        phrase in QUERY_INTERPRETATION_SYSTEM_PROMPT
        for phrase in ["last conversation", "most recent", "latest", "newest"]
    )
    assert has_example
