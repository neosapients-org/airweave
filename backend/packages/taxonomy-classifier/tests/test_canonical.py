"""Generic canonical-taxonomy behaviour on the neutral fixture: load, ids, hierarchy, validator, save/reload, root config."""

import ast
import json

import pytest

from taxonomy_classifier import canonical
from taxonomy_classifier.canonical import load_canonical_taxonomy, load_registry, save_canonical_taxonomy, validate_taxonomy

from .conftest import FIXTURE_TAXONOMY, ROOT


def test_fixture_loads_by_path_with_any_domain_name():
    tax = load_canonical_taxonomy(FIXTURE_TAXONOMY)
    assert tax.domain == "test_fixture" and tax.version == "fixture_v1"
    assert [c["category_id"] for c in tax.categories] == ["A", "A1", "A1x", "B", "C"]
    assert validate_taxonomy(tax.data)["errors"] == []


def test_hierarchy_traversal():
    tax = load_canonical_taxonomy(FIXTURE_TAXONOMY)
    assert tax.parent("A1x") == "A1" and tax.ancestors("A1x") == ["A1", "A"]          # immediate parent first
    assert tax.children("A") == ["A1"] and tax.descendants("A") == ["A1", "A1x"]
    assert set(tax.roots()) == {"A", "B", "C"}
    assert tax.get("A1x")["status"] == "PROBATIONARY"


def test_save_reload_preserves_category_ids(tmp_path):
    tax = load_canonical_taxonomy(FIXTURE_TAXONOMY)
    p = tmp_path / "taxonomy.json"
    save_canonical_taxonomy(tax, p)
    again = load_canonical_taxonomy(p)
    assert [c["category_id"] for c in again.categories] == [c["category_id"] for c in tax.categories]
    assert again.data == tax.data


def test_registry_root_is_configurable_and_domain_is_free_form(tmp_path, monkeypatch):
    dom = tmp_path / "my_custom_corpus"; dom.mkdir()
    (dom / "taxonomy.json").write_text(FIXTURE_TAXONOMY.read_text())
    (tmp_path / "registry.json").write_text(json.dumps({"schema_version": "taxonomy-registry/1", "domains": {
        "my_custom_corpus": {"taxonomy_path": "my_custom_corpus/taxonomy.json"}}}))
    assert load_canonical_taxonomy("my_custom_corpus", root=tmp_path).version == "fixture_v1"
    monkeypatch.setenv(canonical.ROOT_ENV, str(tmp_path))
    assert set(load_registry()["domains"]) == {"my_custom_corpus"}
    with pytest.raises(KeyError):
        load_canonical_taxonomy("finance")                       # nothing is built in: only what the root contains
    monkeypatch.setenv(canonical.ROOT_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        load_registry()


def test_validator_catches_broken_hierarchies():
    base = {"schema_version": "taxonomy-export/1", "families": [], "categories": []}
    cat = lambda i, p, kids, anc, dep: {"category_id": i, "name": i, "description": "d", "status": "ACCEPTED", "domain": "x",
                                        "parent_id": p, "ancestor_ids": anc, "children_ids": kids, "depth": dep}
    cyc = {**base, "categories": [cat("A", "B", ["B"], ["B"], 2), cat("B", "A", ["A"], ["A"], 2)]}
    assert any("cycle" in e for e in validate_taxonomy(cyc)["errors"])
    dangling = {**base, "categories": [cat("A", "Z", [], ["Z"], 2)]}
    assert any("does not exist" in e for e in validate_taxonomy(dangling)["errors"])
    dup = {**base, "categories": [cat("A", None, [], [], 1), cat("A", None, [], [], 1)]}
    assert any("duplicate category_id" in e for e in validate_taxonomy(dup)["errors"])
    bad_status = {**base, "categories": [{**cat("A", None, [], [], 1), "status": "MAYBE"}]}
    assert any("invalid status" in e for e in validate_taxonomy(bad_status)["errors"])
    names = {**base, "categories": [cat("A", None, [], [], 1), {**cat("B", None, [], [], 1), "name": "A"}]}
    r = validate_taxonomy(names)
    assert r["errors"] == [] and any("duplicate name" in w for w in r["warnings"])   # reported, never merged


RESEARCH_MARKERS = ("experiments", "runs/", "final_e2e", "corpus_expansion", "cross_domain_validation", "acceptance_retraining",
                    "category_representation", "taxonomy_pipeline_research", "_work/", "production_taxonomy.pkl")


def test_production_code_does_not_import_or_load_research_code():
    for f in (ROOT / "src").rglob("*.py"):
        src = f.read_text()
        for node in ast.walk(ast.parse(src)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                   [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(m.split(".")[0] in ("experiments", "runs") for m in mods), f
        hits = [m for m in RESEARCH_MARKERS if m in src]
        assert not hits, (f.name, hits)
