"""The run's cost ledger exists from the start, so a run that pays for nothing still reports USD 0."""

import json

from taxonomy_classifier.llm.costs import CostLedger


def test_a_new_ledger_is_written_at_once_with_zero(tmp_path):
    path = tmp_path / "run" / "cost_ledger.json"
    CostLedger(1.0, path)
    assert json.loads(path.read_text()) == {"cap_usd": 1.0, "total_usd": 0.0, "stages": {}, "in_flight_usd": 0.0}


def test_an_existing_ledger_keeps_its_spend_and_takes_the_new_cap(tmp_path):
    path = tmp_path / "cost_ledger.json"
    CostLedger(1.0, path).add("embedding", 0.25)
    CostLedger(2.0, path)
    data = json.loads(path.read_text())
    assert data["total_usd"] == 0.25 and data["cap_usd"] == 2.0 and data["stages"]["embedding"]["calls"] == 1


def test_a_wave_with_its_own_reviewers_still_has_a_ledger(tmp_path, monkeypatch):
    """run_wave writes the wave's ledger even when the caller passes the reviewers (no LLM calls: USD 0)."""
    from types import SimpleNamespace

    import taxonomy_classifier.evolution as evolution
    from taxonomy_classifier import LocalTaxonomyStore, TaxonomyEngine

    monkeypatch.setattr(evolution, "run_wave", lambda *a, **k: SimpleNamespace(new_taxonomy_path=None))
    eng = TaxonomyEngine({"llm": {"max_cost_usd": 1.0}}, store=LocalTaxonomyStore(tmp_path / "store"))
    eng.taxonomy = SimpleNamespace(taxonomy_id="t", version="v1")
    eng.run_wave([], wave=2, new_version="v2", reviewer=object(), structural_reviewer=object())
    ledger = tmp_path / "store" / "t" / "_waves" / "v2" / "cost_ledger.json"
    assert json.loads(ledger.read_text())["total_usd"] == 0.0
