"""Contract on a real (synthetic, offline) build: classify path, topics[], persistence, novelty initialisation."""

import pytest

from taxonomy_classifier.contract import classify
from taxonomy_classifier.novelty.maintenance import run_maintenance
from taxonomy_classifier.taxonomy import artifacts

from .conftest import FakeEmbeddings, FakeLLM, synthetic_chunks
from .test_airweave_integration import built  # noqa: F401  (module-scoped build fixture)

pytestmark = pytest.mark.slow

OFF_TOPIC = ("The cafeteria menu rotates weekly with soups, salads and pasta. Parking passes are issued at reception. "
             "The cafeteria menu rotates weekly with soups, salads and pasta. Parking passes are issued at reception. ") * 6


def test_classify_returns_topics_list_for_every_chunk(built):  # noqa: F811
    chunks = synthetic_chunks(n_groups=2, docs_per_group=1, seed=7, prefix="NEW")
    res = classify(chunks, built["out"], built["cfg"], embed_transport=FakeEmbeddings())
    assert [r["chunk_id"] for r in res.records] == [c["chunk_id"] for c in chunks]
    for r in res.records:
        assert isinstance(r["topics"], list)
        assert r["taxonomy_coverage"]["topic_count"] == len(r["topics"])
        for t in r["topics"]:
            assert t["category_id"] and t["status"] == "accepted" and t["supporting_children"] >= 1
    assert any(r["topics"] for r in res.records)


def test_persistence_reload_keeps_category_ids(built):  # noqa: F811
    a, b = artifacts.load(built["out"]), artifacts.load(built["out"])
    assert [c["category_id"] for c in a.categories] == [c["category_id"] for c in b.categories]
    assert a.assigner.pos_ids == b.assigner.pos_ids


def test_unresolved_children_enter_novelty_pool_and_maintenance_initialises(built, tmp_path):  # noqa: F811
    chunks = [{"chunk_id": "OFF1", "document_id": "DOC_OFF", "text": OFF_TOPIC}]
    res = classify(chunks, built["out"], built["cfg"], embed_transport=FakeEmbeddings())
    assert res.records[0]["topics"] == []
    assert res.unresolved_children and all(u["chunk_id"] == "OFF1" for u in res.unresolved_children)
    summary = run_maintenance(artifacts.load(built["out"]), res.unresolved_children, built["cfg"], tmp_path,
                              log=lambda m: None, embed_transport=FakeEmbeddings(), llm_transport=FakeLLM())
    assert isinstance(summary, dict)
