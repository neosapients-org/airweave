"""Stable topics[] contract: shape, empty / single / multi topics, parent aggregation, coverage, hierarchy.

Uses the neutral fixture taxonomy (tests/fixtures/taxonomy.json); no research taxonomy is needed.
"""

from types import SimpleNamespace

from taxonomy_classifier.canonical import load_canonical_taxonomy
from taxonomy_classifier.contract import CONTRACT_VERSION, to_contract
from taxonomy_classifier.integration.aggregation import aggregate
from taxonomy_classifier.schemas import AirweaveChunk

from .conftest import FIXTURE_TAXONOMY

CFG = {"aggregation": {"min_supporting_children": 1}}
FIX = load_canonical_taxonomy(FIXTURE_TAXONOMY)


class _Tax(SimpleNamespace):
    def info(self, cid):
        c = FIX.get(cid)
        return {"name": c["name"], "broad_family_id": c["broad_family_id"], "broad_family_name": c["broad_family_name"]}


TAX = _Tax(version="fixture_v1", taxonomy_id="fixture")


def child(i, parent, status, cats, source="HYBRID_RRF"):
    return {"taxonomy_chunk_id": f"{parent}::tc{i:03d}", "chunk_id": parent, "status": status, "content_form": "NARRATIVE",
            "char_start": i * 100, "char_end": i * 100 + 100,
            "topics": [{"category_id": c, "source": source, "score": 0.03} for c in cats]}


def run(children, parent="AW1"):
    enriched = aggregate([AirweaveChunk(parent, "DOC1", "text")], children, TAX, CFG)[0]
    return to_contract(enriched, FIX)


def test_zero_topics_is_an_empty_list():
    rec = run([child(0, "AW1", "UNRESOLVED", []), child(1, "AW1", "BYPASS_ADMINISTRATIVE", [])])
    assert rec["contract_version"] == CONTRACT_VERSION
    assert rec["topics"] == [] and rec["status"] == "UNRESOLVED"
    cov = rec["taxonomy_coverage"]
    assert cov["coverage_status"] == "NONE" and cov["taxonomy_children_administrative"] == 1
    assert "category" not in rec and "category_id" not in rec          # no single-category field, ever


def test_record_identifies_the_taxonomy_that_produced_it():
    rec = run([child(0, "AW1", "ASSIGNED", ["B"])])
    assert (rec["taxonomy_id"], rec["taxonomy_version"]) == ("fixture", "fixture_v1")


def test_single_label_appears_as_topics_0():
    rec = run([child(0, "AW1", "ASSIGNED", ["B"]), child(1, "AW1", "ASSIGNED", ["B"])])
    assert len(rec["topics"]) == 1
    t = rec["topics"][0]
    assert t["category_id"] == "B" and t["category_name"] == "Category B" and t["status"] == "accepted"
    assert t["broad_family_id"] == "FAM-B" and t["parent_category_id"] is None and t["category_depth"] == 1
    assert t["supporting_children"] == 2 and t["support_share"] == 1.0 and t["sources"] == ["HYBRID_RRF"]
    assert set(t) >= {"category_id", "category_name", "confidence", "status"}


def test_multiple_children_aggregate_to_parent_with_hierarchy_and_coverage():
    kids = [child(0, "AW1", "ASSIGNED", ["A"]), child(1, "AW1", "ASSIGNED", ["A1x"]),
            child(2, "AW1", "ASSIGNED", ["A1x"]), child(3, "AW1", "UNRESOLVED", [])]
    rec = run(kids)
    by = {t["category_id"]: t for t in rec["topics"]}
    assert set(by) == {"A", "A1x"}
    assert by["A1x"]["supporting_taxonomy_chunks"] == ["AW1::tc001", "AW1::tc002"] and by["A1x"]["support_share"] == 0.5
    assert by["A1x"]["ancestor_category_ids"] == ["A1", "A"] and by["A1x"]["category_depth"] == 3
    assert by["A1x"]["parent_category_name"] == "Category A1"
    cov = rec["taxonomy_coverage"]
    assert (cov["taxonomy_children_total"], cov["taxonomy_children_categorised"], cov["taxonomy_children_uncategorised"],
            cov["child_coverage_ratio"], cov["topic_count"], cov["coverage_status"]) == (4, 3, 1, 0.75, 2, "PARTIAL")
    assert rec["status"] == "PARTIALLY_ASSIGNED" and len(rec["taxonomy_children"]) == 4


def test_ancestors_are_metadata_not_extra_topics():
    rec = run([child(0, "AW1", "ASSIGNED", ["A1x"])])
    assert [t["category_id"] for t in rec["topics"]] == ["A1x"]                   # A1 and A are not assigned
    rec = run([child(0, "AW1", "ASSIGNED", ["A"])])
    assert [t["category_id"] for t in rec["topics"]] == ["A"]                     # children of A are not assigned


def test_full_coverage_and_assignment_only_count():
    rec = run([child(0, "AW1", "ASSIGNED", ["B"]), child(1, "AW1", "ASSIGNED_BY_MEMBERSHIP", ["C"], "DISCOVERY_CLUSTER_MEMBER")])
    cov = rec["taxonomy_coverage"]
    assert cov["coverage_status"] == "FULL" and cov["taxonomy_children_categorised"] == 2
    assert cov["taxonomy_children_categorised_by_assignment_only"] == 1           # membership-only labels excluded


def test_one_child_with_several_categories_is_representable():
    rec = run([child(0, "AW1", "ASSIGNED", ["B", "C"])])          # the multi-label phase needs no schema change
    assert {t["category_id"] for t in rec["topics"]} == {"B", "C"}
    assert rec["taxonomy_coverage"]["taxonomy_children_categorised"] == 1
    assert rec["taxonomy_children"][0]["topic_ids"] == ["B", "C"]


def test_children_only_aggregate_to_their_own_parent():
    kids = [child(0, "AW1", "ASSIGNED", ["B"]), child(0, "AW2", "ASSIGNED", ["C"])]
    recs = [to_contract(e, FIX) for e in aggregate([AirweaveChunk("AW1", "D", "t"), AirweaveChunk("AW2", "D", "t")], kids, TAX, CFG)]
    assert [[t["category_id"] for t in r["topics"]] for r in recs] == [["B"], ["C"]]
