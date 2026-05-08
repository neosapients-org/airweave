"""Unit tests for collection temporal config support."""

from airweave.schemas.search import AirweaveTemporalConfig


def test_collection_has_temporal_config_field():
    """Collection model must have a temporal_config JSON column."""
    from airweave.models.collection import Collection

    assert hasattr(Collection, "temporal_config")


def test_collection_schema_includes_temporal_config():
    """CollectionUpdate must include optional temporal_config."""
    from airweave.schemas.collection import CollectionUpdate

    update = CollectionUpdate(temporal_config=AirweaveTemporalConfig(weight=0.3))
    assert update.temporal_config is not None
    assert update.temporal_config.weight == 0.3
