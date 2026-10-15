"""Filters on chunk fields and OR between filter groups (Airweave A8, contract 5.1 and C8)."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from airweave.core.logging import logger
from airweave.domains.search.adapters.vector_db.filter_translator import FilterTranslator
from airweave.domains.search.builders.search_plan import SearchPlanBuilder
from airweave.domains.search.classic.types import ClassicSearchStrategy
from airweave.domains.search.types.filters import FilterCondition, FilterGroup
from airweave.domains.search.types.plan import RetrievalStrategy, SearchPlan, SearchQuery

# Operators each new field takes (contract 5.1); every other operator is refused.
ALLOWED = {
    "topic_ids": {"contains", "in"},
    "broad_family_ids": {"contains", "in"},
    "taxonomy_status": {"equals", "not_equals", "in", "not_in"},
    "taxonomy_version": {"equals", "not_equals", "in", "not_in"},
    "page_from": {
        "equals",
        "greater_than",
        "less_than",
        "greater_than_or_equal",
        "less_than_or_equal",
    },
    "page_to": {
        "equals",
        "greater_than",
        "less_than",
        "greater_than_or_equal",
        "less_than_or_equal",
    },
    "chunk_char_start": {
        "equals",
        "greater_than",
        "less_than",
        "greater_than_or_equal",
        "less_than_or_equal",
    },
}
OPERATORS = [
    "equals",
    "not_equals",
    "contains",
    "greater_than",
    "less_than",
    "greater_than_or_equal",
    "less_than_or_equal",
    "in",
    "not_in",
]
VALUES = {
    "topic_ids": "acme:credit.loan_credit_quality",
    "broad_family_ids": "acme:credit",
    "taxonomy_status": "ASSIGNED",
    "taxonomy_version": "acme-2026.10",
    "page_from": 12,
    "page_to": 13,
    "chunk_char_start": 48210,
}

# sha256 of the LLM-facing JSON schemas at upstream 370c3e605: the new filter fields must not
# reach the schemas the classic and agentic planners are given.
UPSTREAM_SCHEMA_SHA256 = {
    "SearchPlan": "094eddc138c60de45e3be08155513c775cfac3ac00000e60fa8270933714d533",
    "FilterGroup": "c4d0ad5a286d62f8b78afd0591617f3413e61c0c34ed111bb8f3273d719fad2b",
    "ClassicSearchStrategy": "691b580ad342b3ff676abd3117dd050462c697eec9bbda08f107ed7a4d17cc33",
}


def _cond(field, operator, value) -> FilterCondition:
    return FilterCondition(field=field, operator=operator, value=value)


def _value(field, operator):
    v = VALUES[field]
    return [v] if operator in ("in", "not_in") else v


@pytest.fixture
def translator():
    return FilterTranslator(logger=logger.with_context(request_id="test"))


class TestValidation:
    @pytest.mark.parametrize("field", sorted(ALLOWED))
    @pytest.mark.parametrize("operator", OPERATORS)
    def test_operator_matrix(self, field, operator):
        if operator in ALLOWED[field]:
            assert _cond(field, operator, _value(field, operator)).field.value == field
        else:
            with pytest.raises(ValidationError):
                _cond(field, operator, _value(field, operator))

    @pytest.mark.parametrize("field", ["page_from", "page_to", "chunk_char_start"])
    def test_integer_values(self, field):
        assert _cond(field, "greater_than", "12").value == 12  # numeric string -> int
        for bad in ("12.5", "twelve", True):
            with pytest.raises(ValidationError):
                _cond(field, "equals", bad)

    def test_in_value_with_comma_kept_whole(self):
        cond = _cond("taxonomy_version", "in", ["acme-2026.10,rc1", "globex"])
        assert cond.value == ["acme-2026.10,rc1", "globex"]

    def test_existing_fields_unchanged(self):
        assert _cond("airweave_system_metadata.chunk_index", "greater_than", 3).value == 3
        with pytest.raises(ValidationError):
            _cond("name", "greater_than", "x")


class TestLlmSchemasUnchanged:
    @pytest.mark.parametrize(
        "name,model",
        [
            ("SearchPlan", SearchPlan),
            ("FilterGroup", FilterGroup),
            ("ClassicSearchStrategy", ClassicSearchStrategy),
        ],
    )
    def test_schema_identical_to_upstream(self, name, model):
        raw = json.dumps(model.model_json_schema(), sort_keys=True).encode()
        assert hashlib.sha256(raw).hexdigest() == UPSTREAM_SCHEMA_SHA256[name]

    def test_new_fields_hidden_from_enum(self):
        enum = SearchPlan.model_json_schema()["$defs"]["FilterableField"]["enum"]
        assert not set(ALLOWED) & set(enum)


class TestTranslation:
    @pytest.mark.parametrize(
        "field,operator,value,yql",
        [
            ("topic_ids", "contains", "acme:credit.x", "topic_ids contains 'acme:credit.x'"),
            (
                "broad_family_ids",
                "in",
                ["acme:credit", "acme:revenue"],
                "(broad_family_ids contains 'acme:credit' OR broad_family_ids contains "
                "'acme:revenue')",
            ),
            ("taxonomy_status", "not_equals", "ERROR", "!(taxonomy_status contains 'ERROR')"),
            ("taxonomy_version", "not_in", ["v1"], "(!(taxonomy_version contains 'v1'))"),
            ("page_from", "equals", 12, "page_from = 12"),
            ("page_to", "less_than_or_equal", 13, "page_to <= 13"),
            ("chunk_char_start", "greater_than_or_equal", 48210, "chunk_char_start >= 48210"),
            ("chunk_char_start", "less_than", "49402", "chunk_char_start < 49402"),
        ],
    )
    def test_yql(self, translator, field, operator, value, yql):
        group = FilterGroup(conditions=[_cond(field, operator, value)])
        assert translator.translate([group]) == f"({yql})"

    def test_comma_value_never_split(self, translator):
        group = FilterGroup(conditions=[_cond("taxonomy_version", "in", ["acme-2026.10,rc1"])])
        assert translator.translate([group]) == "((taxonomy_version contains 'acme-2026.10,rc1'))"


def _plan(groups=None) -> SearchPlan:
    return SearchPlan(
        query=SearchQuery(primary="late payments"),
        filter_groups=groups or [],
        limit=10,
        offset=0,
        retrieval_strategy=RetrievalStrategy.HYBRID,
    )


def _group(field, value, operator="equals") -> FilterGroup:
    return FilterGroup(conditions=[_cond(field, operator, value)])


DOCS = _group("airweave_system_metadata.original_entity_id", ["f3a1", "77b0"], "in")
TOPIC = _group("topic_ids", "acme:credit.loan_credit_quality", "contains")


class TestOrGroups:
    def test_default_flattens_as_upstream(self):
        """SEARCH_FILTER_GROUPS_OR off: user groups merged into one AND group (upstream)."""
        result = SearchPlanBuilder.build(_plan(), [DOCS, TOPIC])
        assert len(result.filter_groups) == 1
        assert len(result.filter_groups[0].conditions) == 2

    def test_setting_read_when_not_passed(self, monkeypatch):
        from airweave.core.config import settings

        monkeypatch.setattr(settings, "SEARCH_FILTER_GROUPS_OR", True)
        assert len(SearchPlanBuilder.build(_plan(), [DOCS, TOPIC]).filter_groups) == 2

    def test_or_kept_without_llm_groups(self, translator):
        result = SearchPlanBuilder.build(_plan(), [DOCS, TOPIC], keep_or_groups=True)
        assert [g.model_dump() for g in result.filter_groups] == [
            DOCS.model_dump(),
            TOPIC.model_dump(),
        ]
        yql = translator.translate(result.filter_groups)
        assert yql == (
            "((airweave_system_metadata_original_entity_id contains 'f3a1' OR "
            "airweave_system_metadata_original_entity_id contains '77b0')) OR "
            "(topic_ids contains 'acme:credit.loan_credit_quality')"
        )

    def test_or_kept_with_llm_groups_is_a_cross_product(self):
        llm_a = _group("airweave_system_metadata.source_name", "google_drive")
        llm_b = _group("airweave_system_metadata.source_name", "sharepoint")
        result = SearchPlanBuilder.build(_plan([llm_a, llm_b]), [DOCS, TOPIC], keep_or_groups=True)
        pairs = [[c.field.value for c in g.conditions] for g in result.filter_groups]
        assert len(pairs) == 4
        assert all(p[0] == "airweave_system_metadata.source_name" for p in pairs)
        assert sorted(p[1] for p in pairs) == sorted(
            ["airweave_system_metadata.original_entity_id", "topic_ids"] * 2
        )

    def test_single_group_same_both_ways(self):
        llm = _group("airweave_system_metadata.source_name", "google_drive")
        old = SearchPlanBuilder.build(_plan([llm]), [TOPIC], keep_or_groups=False)
        new = SearchPlanBuilder.build(_plan([llm]), [TOPIC], keep_or_groups=True)
        assert old.model_dump() == new.model_dump()

    def test_or_mode_does_not_mutate_input(self):
        plan = _plan([_group("name", "x")])
        SearchPlanBuilder.build(plan, [DOCS, TOPIC], keep_or_groups=True)
        assert len(plan.filter_groups) == 1 and len(plan.filter_groups[0].conditions) == 1
