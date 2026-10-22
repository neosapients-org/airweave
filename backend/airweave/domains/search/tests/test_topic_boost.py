"""Topic boost in instant search (Airweave A8, contract 5.1, decision D14)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from airweave.adapters.event_bus.fake import FakeEventBus
from airweave.api.context import ApiContext
from airweave.core.logging import logger
from airweave.core.shared_models import AuthMethod
from airweave.domains.access_control.fakes.broker import FakeAccessBroker
from airweave.domains.collections.fakes.repository import FakeCollectionRepository
from airweave.domains.embedders.fakes.embedder import FakeDenseEmbedder, FakeSparseEmbedder
from airweave.domains.embedders.types import DenseEmbedding, SparseEmbedding
from airweave.domains.search.adapters.vector_db.fakes.vector_db import FakeVectorDB
from airweave.domains.search.adapters.vector_db.filter_translator import FilterTranslator
from airweave.domains.search.adapters.vector_db.vespa_client import VespaVectorDB
from airweave.domains.search.executor import SearchPlanExecutor
from airweave.domains.search.fakes.executor import FakeSearchPlanExecutor
from airweave.domains.search.instant.service import InstantSearchService
from airweave.domains.search.types import QueryEmbeddings, SearchResults, TopicBoost
from airweave.domains.search.types.plan import RetrievalStrategy, SearchPlan, SearchQuery
from airweave.domains.source_connections.fakes.repository import FakeSourceConnectionRepository
from airweave.domains.sources.fakes.lifecycle import FakeSourceLifecycleService
from airweave.domains.sources.fakes.registry import FakeSourceRegistry
from airweave.models.collection import Collection
from airweave.schemas.search_v2 import InstantSearchRequest

# The request of contract section 5.1.
CONTRACT_REQUEST = {
    "query": "why did late payments rise in Q3",
    "retrieval_strategy": "hybrid",
    "limit": 20,
    "filter": [
        {
            "conditions": [
                {
                    "field": "airweave_system_metadata.original_entity_id",
                    "operator": "in",
                    "value": ["f3a1", "77b0"],
                }
            ]
        },
        {
            "conditions": [
                {
                    "field": "topic_ids",
                    "operator": "contains",
                    "value": "acme:credit.loan_credit_quality",
                }
            ]
        },
    ],
    "boost": {"topic_ids": ["acme:credit.loan_credit_quality"], "weight": 0.15},
}
BOOST = TopicBoost(topic_ids=["acme:credit.loan_credit_quality", "acme:credit.allowance"])


class TestRequest:
    def test_contract_request_parses(self):
        req = InstantSearchRequest.model_validate(CONTRACT_REQUEST)
        assert req.boost == TopicBoost(topic_ids=["acme:credit.loan_credit_quality"], weight=0.15)
        assert [g.conditions[0].field.value for g in req.filter] == [
            "airweave_system_metadata.original_entity_id",
            "topic_ids",
        ]

    def test_no_boost_by_default(self):
        assert InstantSearchRequest(query="globex").boost is None

    def test_default_weight(self):
        assert TopicBoost(topic_ids=["acme:x"]).weight == 0.15

    @pytest.mark.parametrize("strategy", ["semantic", "keyword"])
    def test_boost_needs_hybrid(self, strategy):
        with pytest.raises(ValidationError, match="hybrid"):
            InstantSearchRequest(query="q", retrieval_strategy=strategy, boost={"topic_ids": []})

    def test_negative_weight_refused(self):
        with pytest.raises(ValidationError):
            TopicBoost(topic_ids=["acme:x"], weight=-0.1)


def _client() -> VespaVectorDB:
    ctx_logger = logger.with_context(request_id="test")
    return VespaVectorDB(
        app=MagicMock(), logger=ctx_logger, filter_translator=FilterTranslator(logger=ctx_logger)
    )


def _plan(strategy: RetrievalStrategy = RetrievalStrategy.HYBRID) -> SearchPlan:
    return SearchPlan(
        query=SearchQuery(primary="late payments"), limit=20, offset=0, retrieval_strategy=strategy
    )


EMBEDDINGS = QueryEmbeddings(
    dense_embeddings=[DenseEmbedding(vector=[0.1, 0.2, 0.3])],
    sparse_embedding=SparseEmbedding(indices=[11, 42], values=[0.5, 1.5]),
)


class TestVespaParams:
    async def test_without_boost_query_is_unchanged(self):
        """No boost: same YQL and params as before the boost existed (contract 5.1)."""
        plain = await _client().compile_query(_plan(), EMBEDDINGS, "col-1")
        explicit = await _client().compile_query(_plan(), EMBEDDINGS, "col-1", boost=None)
        assert plain.raw == explicit.raw
        params = plain.raw["params"]
        assert params["ranking.profile"] == "hybrid"
        assert not {"input.query(topics)", "input.query(topic_weight)"} & set(params)

    async def test_with_boost_uses_hybrid_topic(self):
        plain = await _client().compile_query(_plan(), EMBEDDINGS, "col-1")
        boosted = await _client().compile_query(_plan(), EMBEDDINGS, "col-1", boost=BOOST)
        params = boosted.raw["params"]
        assert params["ranking.profile"] == "hybrid_topic"
        assert params["input.query(topics)"] == {
            "cells": {"acme:credit.loan_credit_quality": 1.0, "acme:credit.allowance": 1.0}
        }
        assert params["input.query(topic_weight)"] == 0.15
        # A boost, not a filter: the YQL and every other parameter are unchanged
        assert boosted.raw["yql"] == plain.raw["yql"]
        rest = {
            k: v
            for k, v in params.items()
            if k not in ("ranking.profile", "input.query(topics)", "input.query(topic_weight)")
        }
        assert rest == {k: v for k, v in plain.raw["params"].items() if k != "ranking.profile"}

    async def test_boost_weight_passed(self):
        boost = TopicBoost(topic_ids=["acme:x"], weight=0.5)
        compiled = await _client().compile_query(_plan(), EMBEDDINGS, "col-1", boost=boost)
        assert compiled.raw["params"]["input.query(topic_weight)"] == 0.5

    async def test_display_leaves_out_tensors(self):
        compiled = await _client().compile_query(_plan(), EMBEDDINGS, "col-1", boost=BOOST)
        assert "input.query(topics)" not in compiled.display


class TestPassThrough:
    async def test_executor_passes_boost_to_vector_db(self):
        vector_db = FakeVectorDB()
        vector_db.seed_results(SearchResults(results=[]))
        executor = SearchPlanExecutor(
            dense_embedder=FakeDenseEmbedder(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_db=vector_db,
            sc_repo=FakeSourceConnectionRepository(),
            source_registry=FakeSourceRegistry(),
            source_lifecycle=FakeSourceLifecycleService(),
            access_broker=FakeAccessBroker(),
        )
        ctx = MagicMock()
        ctx.organization.id = uuid4()
        for boost in (None, BOOST):
            await executor.execute(
                plan=_plan(),
                user_filter=[],
                collection_id="col-1",
                db=AsyncMock(),
                ctx=ctx,
                collection_readable_id="acme",
                boost=boost,
            )
        assert vector_db.boosts == [None, BOOST]

    async def test_instant_search_passes_request_boost(self):
        from airweave.schemas.organization import Organization

        executor, repo = FakeSearchPlanExecutor(), FakeCollectionRepository()
        executor.seed_result(SearchResults(results=[]))
        now = datetime.now(timezone.utc)
        org_id = uuid4()
        col = Collection(
            id=uuid4(),
            name="Acme",
            readable_id="acme",
            organization_id=org_id,
            vector_db_deployment_metadata_id=uuid4(),
        )
        col.created_at = col.modified_at = now
        repo.seed_readable("acme", col)
        ctx = ApiContext(
            request_id="r1",
            organization=Organization(
                id=str(org_id), name="Acme", created_at=now, modified_at=now, enabled_features=[]
            ),
            auth_method=AuthMethod.SYSTEM,
            auth_metadata={},
            logger=logger.with_context(request_id="r1"),
        )
        svc = InstantSearchService(
            executor=executor, collection_repo=repo, event_bus=FakeEventBus()
        )
        await svc.search(AsyncMock(), ctx, "acme", InstantSearchRequest(query="q"))
        await svc.search(
            AsyncMock(), ctx, "acme", InstantSearchRequest.model_validate(CONTRACT_REQUEST)
        )
        assert executor.boosts == [
            None,
            TopicBoost(topic_ids=["acme:credit.loan_credit_quality"], weight=0.15),
        ]


class TestWeightIsBounded:
    @pytest.mark.parametrize("weight", [float("inf"), float("nan"), -0.1, 0.51, 2.0, 1e9])
    def test_a_weight_that_could_swamp_the_hybrid_score_is_refused(self, weight):
        from pydantic import ValidationError

        from airweave.domains.search.types.boost import TopicBoost

        with pytest.raises(ValidationError):
            TopicBoost(topic_ids=["acme:x"], weight=weight)

    @pytest.mark.parametrize("weight", [0.0, 0.15, 0.5])
    def test_the_range_edges_are_allowed(self, weight):
        from airweave.domains.search.types.boost import TopicBoost

        assert TopicBoost(topic_ids=["acme:x"], weight=weight).weight == weight

    def test_json_infinity_is_refused(self):
        from pydantic import ValidationError

        from airweave.domains.search.types.boost import TopicBoost

        with pytest.raises(ValidationError):
            TopicBoost.model_validate_json('{"topic_ids": [], "weight": Infinity}')


def test_the_vespa_topic_match_counts_presence_not_piece_count():
    """A chunk's stored weight is its piece count; the match caps it at 1 and is a share, 0 to 1."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[5]
    for rel in (
        "vespa/app/schemas/base_entity.sd",
        "ops/helm/airweave-svc/files/vespa-app/schemas/base_entity.sd",
    ):
        text = (repo / rel).read_text()
        assert "f(x)(min(x, 1))" in text and "max(sum(query(topics)), 1)" in text, rel
