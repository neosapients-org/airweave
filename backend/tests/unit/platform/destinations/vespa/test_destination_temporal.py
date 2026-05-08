"""Unit tests for Vespa destination temporal support."""

from airweave.schemas.search import AirweaveTemporalConfig


def test_vespa_destination_supports_temporal_relevance():
    """VespaDestination must declare temporal relevance support."""
    from airweave.platform.destinations.vespa.destination import VespaDestination

    assert VespaDestination.supports_temporal_relevance is True


def test_translate_temporal_returns_params():
    """translate_temporal must return Vespa ranking params."""
    from airweave.platform.destinations.vespa.destination import VespaDestination

    dest = VespaDestination()
    config = AirweaveTemporalConfig(weight=0.4, reference_field="updated_at")
    result = dest.translate_temporal(config)

    assert result is not None
    assert result["freshness_weight"] == 0.4
    assert result["freshness_field"] == "updated_at"


def test_translate_temporal_none_returns_none():
    """translate_temporal(None) must return None."""
    from airweave.platform.destinations.vespa.destination import VespaDestination

    dest = VespaDestination()
    assert dest.translate_temporal(None) is None
