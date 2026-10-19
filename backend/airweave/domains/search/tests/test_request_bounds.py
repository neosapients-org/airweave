"""Request-boundary limits: filter groups, conditions, list values, offsets, topic ids, merges."""

import pytest
from pydantic import ValidationError

from airweave.core.exceptions import InvalidInputError
from airweave.domains.search.builders.search_plan import MAX_MERGED_GROUPS, SearchPlanBuilder
from airweave.domains.search.types import (
    FilterGroup,
    RetrievalStrategy,
    SearchPlan,
    SearchQuery,
    TopicBoost,
)
from airweave.domains.search.types.filters import MAX_CHUNK_FIELD_LIST_VALUES, FilterCondition
from airweave.schemas.search_v2 import (
    MAX_BROWSE_OFFSET,
    MAX_CONDITIONS_PER_GROUP,
    MAX_FILTER_GROUPS,
    BrowseRequest,
    InstantSearchRequest,
)


def group(n_conditions: int = 1) -> dict:
    cond = {"field": "name", "operator": "equals", "value": "x"}
    return {"conditions": [cond] * n_conditions}


class TestFilterBounds:
    def test_too_many_groups_are_refused_on_every_request_shape(self):
        for build in (
            lambda g: BrowseRequest(filter=g),
            lambda g: InstantSearchRequest(query="q", filter=g),
        ):
            build([group()] * MAX_FILTER_GROUPS)
            with pytest.raises(ValidationError):
                build([group()] * (MAX_FILTER_GROUPS + 1))

    def test_too_many_conditions_in_a_group_are_refused(self):
        BrowseRequest(filter=[group(MAX_CONDITIONS_PER_GROUP)])
        with pytest.raises(ValidationError):
            BrowseRequest(filter=[group(MAX_CONDITIONS_PER_GROUP + 1)])

    def test_a_deep_browse_offset_is_refused(self):
        BrowseRequest(offset=MAX_BROWSE_OFFSET)
        with pytest.raises(ValidationError):
            BrowseRequest(offset=MAX_BROWSE_OFFSET + 1)

    def test_a_long_list_on_a_taxonomy_field_is_refused(self):
        def cond(n):
            return FilterCondition(
                field="topic_ids", operator="in", value=[f"acme:t{i}" for i in range(n)]
            )

        cond(MAX_CHUNK_FIELD_LIST_VALUES)
        with pytest.raises(ValidationError):
            cond(MAX_CHUNK_FIELD_LIST_VALUES + 1)


class TestTopicIds:
    def test_registry_ids_are_accepted(self):
        TopicBoost(topic_ids=["acme:credit.loan_quality", "acme-2:a_b"])

    @pytest.mark.parametrize("bad", ["x" * 201, "", "a b", "a]", 'a"b', ":lead"])
    def test_a_long_or_malformed_id_is_refused(self, bad):
        with pytest.raises(ValidationError):
            TopicBoost(topic_ids=[bad])


class TestMergedGroupCap:
    def plan(self, n):
        return SearchPlan(
            query=SearchQuery(primary="q"),
            filter_groups=[FilterGroup(**group()) for _ in range(n)],
            limit=10,
            offset=0,
            retrieval_strategy=RetrievalStrategy.HYBRID,
        )

    def test_the_product_over_the_cap_is_a_client_error(self):
        users = [FilterGroup(**group()) for _ in range(MAX_FILTER_GROUPS)]
        n = MAX_MERGED_GROUPS // MAX_FILTER_GROUPS
        SearchPlanBuilder.build(self.plan(n), users, keep_or_groups=True)
        with pytest.raises(InvalidInputError):
            SearchPlanBuilder.build(self.plan(n + 1), users, keep_or_groups=True)
