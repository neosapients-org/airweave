from types import SimpleNamespace

from taxonomy_classifier.integration.aggregation import aggregate
from taxonomy_classifier.integration.lineage import chunk_status, parent_of
from taxonomy_classifier.schemas import AirweaveChunk


class FakeTaxonomy:
    version = "t1"
    INFO = {"CAT001": ("Deferred Taxes", "BF01", "Tax & Accounting Estimates"),
            "CAT002": ("Fair Value Measurement", "BF02", "Fair Value, Securities & Derivatives"),
            "CAT003": ("Debt", "BF03", "Funding")}

    def info(self, cid):
        n, f, fn = self.INFO[cid]
        return {"name": n, "broad_family_id": f, "broad_family_name": fn}


def child(tid, status, cats, form="NARRATIVE"):
    return {"taxonomy_chunk_id": tid, "chunk_id": parent_of(tid), "status": status, "char_start": 0, "char_end": 1,
            "content_form": form, "topics": [{"category_id": c, "source": "HYBRID_RRF", "score": 0.03} for c in cats]}


def test_parent_of():
    assert parent_of("AW_012::tc003") == "AW_012"
    assert parent_of("a::b::tc000") == "a::b"


def test_chunk_status_rules():
    assert chunk_status(["ASSIGNED", "ASSIGNED"], True) == "ASSIGNED"
    assert chunk_status(["ASSIGNED", "UNRESOLVED"], True) == "PARTIALLY_ASSIGNED"
    assert chunk_status(["ASSIGNED", "BYPASS_ADMINISTRATIVE"], True) == "ASSIGNED"
    assert chunk_status(["UNRESOLVED", "BYPASS_ADMINISTRATIVE"], False) == "UNRESOLVED"
    assert chunk_status(["BYPASS_ADMINISTRATIVE", "REJECT_NONSEMANTIC"], False) == "ADMINISTRATIVE"
    assert chunk_status([], False) == "ADMINISTRATIVE"


def test_aggregation_worked_example():
    """README example: AW_012 -> TC_001 Deferred Taxes; TC_002 Deferred Taxes + Fair Value."""
    chunks = [AirweaveChunk("AW_012", "DOC_01", "...", {"company": "X"}), AirweaveChunk("AW_013", "DOC_01", "...")]
    kids = [child("AW_012::tc000", "ASSIGNED", ["CAT001"]), child("AW_012::tc001", "ASSIGNED", ["CAT001", "CAT002"]),
            child("AW_013::tc000", "BYPASS_ADMINISTRATIVE", [], "SIGNATURE")]
    out = aggregate(chunks, kids, FakeTaxonomy(), {"aggregation": {"min_supporting_children": 1}})
    a, b = out
    assert a["chunk_id"] == "AW_012" and a["metadata"] == {"company": "X"}
    tx = a["taxonomy"]
    assert tx["status"] == "ASSIGNED"
    assert tx["topic_names"] == ["Deferred Taxes", "Fair Value Measurement"]
    assert tx["topics"][0]["supporting_taxonomy_chunks"] == ["AW_012::tc000", "AW_012::tc001"]
    assert tx["topics"][1]["supporting_taxonomy_chunks"] == ["AW_012::tc001"]
    assert tx["broad_family_ids"] == ["BF01", "BF02"]
    assert "content_form" not in tx and a["content_metadata"]["content_form"] == "NARRATIVE"   # never mixed into topics
    assert b["taxonomy"]["status"] == "ADMINISTRATIVE" and b["taxonomy"]["topics"] == []


def test_min_supporting_children():
    chunks = [AirweaveChunk("AW_1", "D", "...")]
    kids = [child("AW_1::tc000", "ASSIGNED", ["CAT001"]), child("AW_1::tc001", "ASSIGNED", ["CAT001", "CAT003"])]
    out = aggregate(chunks, kids, FakeTaxonomy(), {"aggregation": {"min_supporting_children": 2}})
    assert out[0]["taxonomy"]["topic_ids"] == ["CAT001"]
