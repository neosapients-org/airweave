"""Unit tests for QueryBuilder temporal ranking params."""

from airweave.platform.destinations.vespa.query_builder import QueryBuilder


def test_build_params_without_temporal_uses_hybrid_profile():
    """Without temporal config, ranking profile should remain hybrid."""
    qb = QueryBuilder()
    params = qb.build_params(
        queries=["test query"],
        limit=10,
        offset=0,
        dense_embeddings=[[0.1] * 3072],
        temporal_params=None,
    )

    assert params["ranking.profile"] == "hybrid"


def test_build_params_with_temporal_uses_freshness_profile():
    """Temporal config should switch to freshness rank profile and features."""
    qb = QueryBuilder()
    params = qb.build_params(
        queries=["test query"],
        limit=10,
        offset=0,
        dense_embeddings=[[0.1] * 3072],
        temporal_params={"freshness_weight": 0.4, "freshness_field": "updated_at"},
    )

    assert params["ranking.profile"] == "freshness_hybrid"
    assert params["ranking.features.query(freshness_weight)"] == 0.4
    assert params["ranking.features.query(freshness_field)"] == "updated_at"
