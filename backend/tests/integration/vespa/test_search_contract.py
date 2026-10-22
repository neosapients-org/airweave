"""Search against Vespa with the new fields: filters, OR groups, topic boost (Airweave A8).

Contract airweave-neo/1, section 5.1. See conftest.py: needs a disposable Vespa, skipped
otherwise. Embeddings are tiny fake vectors; no provider is called.
"""

from __future__ import annotations

import pytest

from airweave.domains.embedders.types import DenseEmbedding, SparseEmbedding
from airweave.domains.search.builders.search_plan import SearchPlanBuilder
from airweave.domains.search.types import QueryEmbeddings, TopicBoost
from airweave.domains.search.types.filters import FilterCondition, FilterGroup
from airweave.domains.search.types.plan import RetrievalStrategy, SearchPlan, SearchQuery
from airweave.platform.destinations.vespa.types import VespaDocument

LOAN, REVENUE = "acme:credit.loan_credit_quality", "acme:revenue.revenue_growth"

# (id, chunk fields); every chunk also gets text, metadata and vectors in _doc()
CHUNKS = {
    "f3a1__chunk_0": {"topic_ids": {LOAN: 1}, "broad_family_ids": ["acme:credit"],
                      "taxonomy_status": "ASSIGNED", "taxonomy_version": "acme-2026.10,rc1",
                      "page_from": 1, "page_to": 1, "chunk_char_start": 0},
    "77b0__chunk_0": {"topic_ids": {REVENUE: 1}, "broad_family_ids": ["acme:revenue"],
                      "taxonomy_status": "ASSIGNED", "taxonomy_version": "acme-2026.10",
                      "page_from": 2, "page_to": 3, "chunk_char_start": 1000},
    "9c00__chunk_0": {},  # written with the upstream settings: no new fields at all
    "f3a1__chunk_1": {"taxonomy_status": "ERROR", "taxonomy_version": "acme-2026.10",
                      "page_from": 5, "page_to": 5, "chunk_char_start": 5000},
}


def _doc(entity_id: str, collection_id: str, fields: dict, rank: int, dim: int) -> VespaDocument:
    original = entity_id.split("__")[0]
    return VespaDocument(
        schema="base_entity",
        id=f"BaseEntity_{entity_id}",
        fields={
            "entity_id": entity_id,
            "name": f"{original}.pdf",
            "textual_representation": "late payments rose in the quarter " * (4 - rank),
            "airweave_system_metadata_collection_id": collection_id,
            "airweave_system_metadata_original_entity_id": original,
            "airweave_system_metadata_source_name": "stub",
            "airweave_system_metadata_entity_type": "BaseEntity",
            "airweave_system_metadata_chunk_index": int(entity_id.rsplit("_", 1)[1]),
            "payload": '{"web_url": "stub://acme"}',
            "access_is_public": True,
            "access_viewers": [],
            "dense_embedding": {"values": [0.1 * (rank + 1)] * dim},
            "sparse_embedding": {"cells": [{"address": {"token": "7"}, "value": 1.0 + rank}]},
            **fields,
        },
    )


@pytest.fixture
def corpus(collection_id, feed, embedding_dim):
    feed([_doc(eid, collection_id, f, rank, embedding_dim)
          for rank, (eid, f) in enumerate(CHUNKS.items())])
    return collection_id


def _cond(field, operator, value):
    return FilterCondition(field=field, operator=operator, value=value)


async def _ids(vector_db, collection_id, groups) -> set:
    results = await vector_db.filter_search(groups, collection_id, limit=50)
    return {r.entity_id for r in results}


@pytest.mark.parametrize(
    "field,operator,value,expected",
    [
        ("topic_ids", "contains", LOAN, {"f3a1__chunk_0"}),
        ("topic_ids", "in", [LOAN, REVENUE], {"f3a1__chunk_0", "77b0__chunk_0"}),
        ("broad_family_ids", "contains", "acme:revenue", {"77b0__chunk_0"}),
        ("broad_family_ids", "in", ["acme:credit"], {"f3a1__chunk_0"}),
        ("taxonomy_status", "equals", "ERROR", {"f3a1__chunk_1"}),
        ("taxonomy_status", "not_equals", "ERROR",
         {"f3a1__chunk_0", "77b0__chunk_0", "9c00__chunk_0"}),
        ("taxonomy_status", "in", ["ASSIGNED"], {"f3a1__chunk_0", "77b0__chunk_0"}),
        ("taxonomy_status", "not_in", ["ASSIGNED", "ERROR"], {"9c00__chunk_0"}),
        ("taxonomy_version", "equals", "acme-2026.10", {"77b0__chunk_0", "f3a1__chunk_1"}),
        ("taxonomy_version", "in", ["acme-2026.10,rc1"], {"f3a1__chunk_0"}),  # comma kept
        ("taxonomy_version", "not_in", ["acme-2026.10,rc1"],
         {"77b0__chunk_0", "9c00__chunk_0", "f3a1__chunk_1"}),
        ("page_from", "equals", 2, {"77b0__chunk_0"}),
        ("page_from", "greater_than", 1, {"77b0__chunk_0", "f3a1__chunk_1"}),
        ("page_to", "less_than_or_equal", 3, {"f3a1__chunk_0", "77b0__chunk_0"}),
        ("page_to", "less_than", 3, {"f3a1__chunk_0"}),
        ("chunk_char_start", "greater_than_or_equal", 1000, {"77b0__chunk_0", "f3a1__chunk_1"}),
        ("chunk_char_start", "less_than", 1000, {"f3a1__chunk_0"}),
    ],
)
async def test_each_field_with_each_operator(corpus, vector_db, field, operator, value, expected):
    groups = [FilterGroup(conditions=[_cond(field, operator, value)])]
    assert await _ids(vector_db, corpus, groups) == expected


async def test_range_on_chunk_char_start(corpus, vector_db):
    groups = [FilterGroup(conditions=[
        _cond("chunk_char_start", "greater_than", 0),
        _cond("chunk_char_start", "less_than_or_equal", 1000),
    ])]
    assert await _ids(vector_db, corpus, groups) == {"77b0__chunk_0"}


async def test_or_groups_return_the_union(corpus, vector_db):
    docs = FilterGroup(conditions=[
        _cond("airweave_system_metadata.original_entity_id", "in", ["9c00"])])
    topic = FilterGroup(conditions=[_cond("topic_ids", "contains", REVENUE)])
    plan = SearchPlan(query=SearchQuery(primary="x"), limit=10, offset=0,
                      retrieval_strategy=RetrievalStrategy.HYBRID)
    with_or = SearchPlanBuilder.build(plan, [docs, topic], keep_or_groups=True)
    assert await _ids(vector_db, corpus, with_or.filter_groups) == {"9c00__chunk_0",
                                                                    "77b0__chunk_0"}
    flattened = SearchPlanBuilder.build(plan, [docs, topic], keep_or_groups=False)
    assert await _ids(vector_db, corpus, flattened.filter_groups) == set()  # upstream: AND


async def _hybrid(vector_db, collection_id, boost=None, dim: int = 8) -> list:
    plan = SearchPlan(query=SearchQuery(primary="late payments"), limit=10, offset=0,
                      retrieval_strategy=RetrievalStrategy.HYBRID)
    embeddings = QueryEmbeddings(
        dense_embeddings=[DenseEmbedding(vector=[0.2] * dim)],
        sparse_embedding=SparseEmbedding(indices=[7], values=[1.0]),
    )
    compiled = await vector_db.compile_query(plan, embeddings, collection_id, boost=boost)
    return (await vector_db.execute_query(compiled)).results


async def test_boost_moves_labelled_chunks_up_and_keeps_the_rest(corpus, vector_db):
    plain = await _hybrid(vector_db, corpus)
    boosted = await _hybrid(vector_db, corpus, TopicBoost(topic_ids=[REVENUE], weight=5.0))
    plain_ids, boosted_ids = [r.entity_id for r in plain], [r.entity_id for r in boosted]
    assert set(boosted_ids) == set(plain_ids) == set(CHUNKS)  # a boost, not a filter
    assert boosted_ids[0] == "77b0__chunk_0"
    assert boosted_ids.index("77b0__chunk_0") <= plain_ids.index("77b0__chunk_0")


async def test_zero_weight_ranks_as_hybrid(corpus, vector_db):
    plain = await _hybrid(vector_db, corpus)
    zero = await _hybrid(vector_db, corpus, TopicBoost(topic_ids=[LOAN], weight=0.0))
    assert [r.entity_id for r in zero] == [r.entity_id for r in plain]
    assert [round(r.relevance_score, 6) for r in zero] == [
        round(r.relevance_score, 6) for r in plain
    ]


async def test_results_carry_the_new_fields(corpus, vector_db):
    by_id = {r.entity_id: r for r in await _hybrid(vector_db, corpus)}
    assert by_id["f3a1__chunk_0"].topic_ids == {LOAN: 1}
    assert by_id["77b0__chunk_0"].page_to == 3
    assert by_id["9c00__chunk_0"].topic_ids is None
