"""Classifier records to contract fields (plan A7 mapping.py; contract 4.2, 4.3)."""

import json
import re

import pytest

from airweave.domains.taxonomy.mapping import (
    ADMINISTRATIVE,
    ASSIGNED,
    PARTIALLY_ASSIGNED,
    UNRESOLVED,
    RegistryMap,
    RegistryMapError,
    administrative_labels,
    map_record,
)

REGISTRY = RegistryMap(
    "acme-2026.10",
    {"CAT069": "acme:credit.loan_credit_quality", "CAT070": "acme:credit.allowance_for_losses",
     "CAT101": "acme:revenue.revenue_growth"},
    {"F6": "acme:credit", "F9": "acme:revenue"},
)
FAMILY = {"CAT069": "F6", "CAT070": "F6", "CAT101": "F9", "CAT999": "F6"}
BUILD_ID = re.compile(r"\b(CAT\d+|CAT_NEW_\d+|BF\d+|F\d+)\b")


def _record(pieces, topics=None):
    """A topics/1 record: pieces as (status, start, end, [build ids])."""
    children = [{"taxonomy_chunk_id": f"t#{i}", "status": s, "char_start": a, "char_end": b,
                 "topic_ids": ids} for i, (s, a, b, ids) in enumerate(pieces)]
    if topics is None:
        support: dict = {}
        for c in children:
            for cid in c["topic_ids"]:
                support[cid] = support.get(cid, 0) + 1
        topics = [{"category_id": cid, "broad_family_id": FAMILY.get(cid), "supporting_children": n}
                  for cid, n in support.items()]
    return {"contract_version": "topics/1", "status": "ASSIGNED", "topics": topics,
            "taxonomy_children": children}


class TestContractExample:
    def test_record_of_section_4_3(self):
        """One 300-token piece covering the own text: the span of contract 4.3/4.4."""
        rec = _record([("ASSIGNED", 0, 1011, ["CAT069"])])
        labels = map_record(rec, REGISTRY, own_start=48391, own_end=49402)
        assert labels.topic_ids == {"acme:credit.loan_credit_quality": 1}
        assert labels.broad_family_ids == ["acme:credit"]
        assert labels.taxonomy_spans == (
            '[{"c":"acme:credit.loan_credit_quality","s":48391,"e":49402,"p":[0,0]}]'
        )
        assert (labels.taxonomy_status, labels.taxonomy_piece_count) == (ASSIGNED, 1)
        assert (labels.taxonomy_admin_pieces, labels.taxonomy_unresolved_pieces) == (0, 0)

    def test_two_pieces_two_categories(self):
        """The 600-token fallback: spans differ per piece (contract 4.3)."""
        rec = _record([("ASSIGNED", 0, 500, ["CAT069"]), ("ASSIGNED", 500, 990, ["CAT101"])])
        labels = map_record(rec, REGISTRY, own_start=1000)
        spans = json.loads(labels.taxonomy_spans)
        assert spans == [
            {"c": "acme:credit.loan_credit_quality", "s": 1000, "e": 1500, "p": [0, 0]},
            {"c": "acme:revenue.revenue_growth", "s": 1500, "e": 1990, "p": [1, 1]},
        ]
        assert labels.topic_ids == {"acme:credit.loan_credit_quality": 1,
                                    "acme:revenue.revenue_growth": 1}
        assert labels.broad_family_ids == ["acme:credit", "acme:revenue"]

    def test_run_of_pieces_is_one_span_with_weight_two(self):
        rec = _record([("ASSIGNED", 0, 400, ["CAT069"]), ("ASSIGNED", 400, 900, ["CAT069"])])
        labels = map_record(rec, REGISTRY, own_start=10)
        assert labels.topic_ids == {"acme:credit.loan_credit_quality": 2}
        assert json.loads(labels.taxonomy_spans) == [
            {"c": "acme:credit.loan_credit_quality", "s": 10, "e": 910, "p": [0, 1]}]


class TestTranslation:
    def test_missing_category_dropped_and_counted_unresolved(self):
        rec = _record([("ASSIGNED", 0, 300, ["CAT999"]), ("ASSIGNED", 300, 600, ["CAT069"])])
        labels = map_record(rec, REGISTRY, own_start=0)
        assert labels.topic_ids == {"acme:credit.loan_credit_quality": 1}
        assert labels.taxonomy_status == PARTIALLY_ASSIGNED
        assert labels.taxonomy_unresolved_pieces == 1
        assert labels.untranslated == ["CAT999"]
        assert "CAT999" not in labels.taxonomy_spans

    def test_only_label_missing_gives_unresolved(self):
        labels = map_record(_record([("ASSIGNED", 0, 300, ["CAT999"])]), REGISTRY, own_start=0)
        assert (labels.taxonomy_status, labels.topic_ids, labels.taxonomy_spans) == (
            UNRESOLVED, {}, "[]")

    def test_missing_family_dropped_category_kept(self):
        rec = _record([("ASSIGNED", 0, 300, ["CAT069"])],
                      topics=[{"category_id": "CAT069", "broad_family_id": "F77",
                               "supporting_children": 1}])
        labels = map_record(rec, REGISTRY, own_start=0)
        assert labels.topic_ids == {"acme:credit.loan_credit_quality": 1}
        assert labels.broad_family_ids == [] and labels.untranslated == ["F77"]

    def test_no_build_id_ever_written(self):
        rec = _record([("ASSIGNED", 0, 200, ["CAT069", "CAT999"]),
                       ("ASSIGNED", 200, 400, ["CAT101"])])
        labels = map_record(rec, REGISTRY, own_start=0)
        written = json.dumps([labels.topic_ids, labels.broad_family_ids, labels.taxonomy_spans])
        assert not BUILD_ID.search(written)


class TestStatusesAndCounts:
    @pytest.mark.parametrize("admin_status", ["BYPASS_ADMINISTRATIVE", "REJECT_NONSEMANTIC"])
    def test_administrative_only(self, admin_status):
        labels = map_record(_record([(admin_status, 0, 50, [])]), REGISTRY, own_start=0)
        assert (labels.taxonomy_status, labels.taxonomy_admin_pieces) == (ADMINISTRATIVE, 1)

    def test_partial_counts(self):
        rec = _record([("ASSIGNED", 0, 100, ["CAT069"]), ("UNRESOLVED", 100, 200, []),
                       ("BYPASS_ADMINISTRATIVE", 200, 230, [])])
        labels = map_record(rec, REGISTRY, own_start=0)
        assert labels.taxonomy_status == PARTIALLY_ASSIGNED
        assert (labels.taxonomy_piece_count, labels.taxonomy_admin_pieces,
                labels.taxonomy_unresolved_pieces) == (3, 1, 1)

    def test_spans_kept_inside_own_range(self):
        labels = map_record(_record([("ASSIGNED", 0, 120, ["CAT069"])]), REGISTRY,
                            own_start=10, own_end=100)
        (span,) = json.loads(labels.taxonomy_spans)
        assert (span["s"], span["e"]) == (10, 100)

    def test_empty_own_text(self):
        labels = administrative_labels()
        assert labels.taxonomy_status == ADMINISTRATIVE and labels.taxonomy_piece_count == 0


class TestRegistryMap:
    def test_from_json(self):
        rm = RegistryMap.from_json({"taxonomy_version": "v1", "categories": {"CAT1": "a:b.c"},
                                    "families": {"F1": "a:b"}})
        assert rm.categories == {"CAT1": "a:b.c"}

    @pytest.mark.parametrize("bad", [{}, {"taxonomy_version": "v", "categories": [], "families": {}},
                                     {"taxonomy_version": "v", "categories": {"A": 1}, "families": {}}])
    def test_malformed_refused(self, bad):
        with pytest.raises(RegistryMapError):
            RegistryMap.from_json(bad)


class TestSharedFixture:
    """The shared fixture's labelled chunks, rebuilt from one-piece classifier records."""

    def test_every_chunk_maps_to_its_fixture_fields(self, airweave_neo_chunk_fixture):
        fx = airweave_neo_chunk_fixture
        reg_ids = sorted({c for ch in fx["chunks"] for c in ch["topic_ids"]})
        build = {r: f"CAT{i:03d}" for i, r in enumerate(reg_ids, start=1)}
        fams = sorted({r.split(".")[0] for r in reg_ids})
        fam_build = {f: f"BF{i:02d}" for i, f in enumerate(fams, start=1)}
        registry = RegistryMap("acme-2026.10", {b: r for r, b in build.items()},
                               {b: f for f, b in fam_build.items()})
        for ch in fx["chunks"]:
            own_len = ch["chunk_char_end"] - ch["chunk_own_start"]
            status = {"ADMINISTRATIVE": "BYPASS_ADMINISTRATIVE", "UNRESOLVED": "UNRESOLVED"}.get(
                ch["taxonomy_status"], "ASSIGNED")
            ids = [build[r] for r in ch["topic_ids"]]
            topics = [{"category_id": build[r], "broad_family_id": fam_build[r.split(".")[0]],
                       "supporting_children": 1} for r in ch["topic_ids"]]
            rec = _record([(status, 0, own_len, ids)], topics=topics)
            labels = map_record(rec, registry, ch["chunk_own_start"], ch["chunk_char_end"])
            assert labels.topic_ids == ch["topic_ids"], ch["entity_id"]
            assert labels.broad_family_ids == ch["broad_family_ids"]
            assert labels.taxonomy_spans == ch["taxonomy_spans"]
            assert labels.taxonomy_status == ch["taxonomy_status"]
            assert (labels.taxonomy_piece_count, labels.taxonomy_admin_pieces,
                    labels.taxonomy_unresolved_pieces) == (
                ch["taxonomy_piece_count"], ch["taxonomy_admin_pieces"],
                ch["taxonomy_unresolved_pieces"])
