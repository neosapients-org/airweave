"""Unit tests for temporal extensions in query interpretation."""

from airweave.search.operations.query_interpretation import ExtractedFilters


def test_extracted_filters_has_temporal_weight_field():
    """ExtractedFilters schema must include an optional temporal_weight field."""
    schema = ExtractedFilters.model_json_schema()
    assert "temporal_weight" in schema.get("properties", {})


def test_extracted_filters_temporal_weight_defaults_none():
    """temporal_weight must default to None when not provided by LLM."""
    result = ExtractedFilters(filters=[], confidence=0.9)
    assert result.temporal_weight is None


def test_extracted_filters_temporal_weight_accepts_float():
    """temporal_weight must accept float values between 0 and 1."""
    result = ExtractedFilters(filters=[], confidence=0.9, temporal_weight=0.7)
    assert result.temporal_weight == 0.7
