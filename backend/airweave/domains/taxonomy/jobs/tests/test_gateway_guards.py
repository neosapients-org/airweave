"""The gateway's guards that need no classifier package: ledgers, artifact names, a wave's bill."""

import json
import sys
import types
from pathlib import Path

import pytest

from airweave.domains.taxonomy.jobs.gateway import (
    CallLogDamaged,
    CostUnknown,
    PackageGateway,
    SpendCapReached,
    _ledger_total,
    refuse_if_capped,
)
from airweave.domains.taxonomy.jobs.tests.helpers import NEW, taxonomy


class TestLedger:
    def test_the_total_is_read(self, tmp_path):
        ledger = tmp_path / "cost_ledger.json"
        ledger.write_text(json.dumps({"total_usd": 0.42}))
        assert _ledger_total(ledger) == 0.42

    def test_a_missing_ledger_is_unknown_not_zero(self, tmp_path):
        with pytest.raises(CostUnknown, match="missing or unreadable"):
            _ledger_total(tmp_path / "cost_ledger.json")

    @pytest.mark.parametrize(
        "content", ["not json", "{}", '{"total_usd": "lots"}', '{"total_usd": null}', "[]"]
    )
    def test_a_malformed_ledger_is_unknown_not_zero(self, tmp_path, content):
        ledger = tmp_path / "cost_ledger.json"
        ledger.write_text(content)
        with pytest.raises(CostUnknown):
            _ledger_total(ledger)

    @pytest.mark.parametrize("total", [-1.0, float("inf"), float("nan")])
    def test_an_impossible_total_is_unknown_too(self, tmp_path, total):
        ledger = tmp_path / "cost_ledger.json"
        ledger.write_text(json.dumps({"total_usd": total}))
        with pytest.raises(CostUnknown, match="impossible"):
            _ledger_total(ledger)

    def test_zero_is_a_total_when_the_ledger_says_so(self, tmp_path):
        ledger = tmp_path / "cost_ledger.json"
        ledger.write_text(json.dumps({"total_usd": 0.0}))
        assert _ledger_total(ledger) == 0.0


def _calls(tmp_path, *statuses):
    (tmp_path / "llm_calls.jsonl").write_text(
        "".join(json.dumps({"stage": "review", "status": s}) + "\n" for s in statuses)
    )


class TestRefusedCalls:
    def test_a_run_whose_calls_were_all_answered_passes(self, tmp_path):
        _calls(tmp_path, "OK", "OK", "FAILED")
        refuse_if_capped(tmp_path)

    def test_a_run_without_model_calls_passes(self, tmp_path):
        refuse_if_capped(tmp_path)

    def test_one_refused_call_fails_the_job_naming_the_cap(self, tmp_path):
        _calls(tmp_path, "OK", "BUDGET_EXCEEDED", "OK", "BUDGET_EXCEEDED")
        with pytest.raises(SpendCapReached, match="refused 2 model call"):
            refuse_if_capped(tmp_path)

    def test_the_package_error_it_explains_is_kept_as_the_cause(self, tmp_path):
        _calls(tmp_path, "BUDGET_EXCEEDED")
        cause = RuntimeError("consolidation failed: BUDGET_EXCEEDED")
        with pytest.raises(SpendCapReached) as e:
            refuse_if_capped(tmp_path, cause)
        assert e.value.__cause__ is cause

    @pytest.mark.parametrize("damaged", ['{"stage": "review", "status": "BUDG', "[1]"])
    def test_an_unreadable_record_fails_the_job_as_it_may_hide_a_refusal(self, tmp_path, damaged):
        (tmp_path / "llm_calls.jsonl").write_text(f'{{"status": "OK"}}\n{damaged}\n')
        with pytest.raises(CallLogDamaged, match="1 record.*cannot be read"):
            refuse_if_capped(tmp_path)

    def test_blank_lines_are_not_records(self, tmp_path):
        (tmp_path / "llm_calls.jsonl").write_text('{"status": "OK"}\n\n')
        refuse_if_capped(tmp_path)


class TestAllowedFiles:
    def folder(self, tmp_path):
        folder = tmp_path / "build"
        folder.mkdir()
        (folder / "final_taxonomy.json").write_text("{}")
        (folder / "dense_prototypes.npy").write_bytes(b"npy")
        return folder

    def test_listed_plain_files_are_returned(self, tmp_path):
        folder = self.folder(tmp_path)
        files = PackageGateway._allowed_files(folder, {"artifacts": {"dense_prototypes.npy": "x"}})
        assert sorted(files) == ["dense_prototypes.npy", "final_taxonomy.json"]

    @pytest.mark.parametrize(
        "name", ["../secret", "../../etc/passwd", "/etc/passwd", "sub/file", "..", ".", ".env"]
    )
    def test_a_name_that_is_not_a_plain_file_name_is_refused(self, tmp_path, name):
        folder = self.folder(tmp_path)
        (tmp_path / "secret").write_text("customer data")
        with pytest.raises(ValueError, match="plain file name"):
            PackageGateway._allowed_files(folder, {"artifacts": {name: "x"}})

    def test_a_listed_file_that_is_not_there_is_refused(self, tmp_path):
        folder = self.folder(tmp_path)
        with pytest.raises(ValueError, match="not a file of the build folder"):
            PackageGateway._allowed_files(folder, {"artifacts": {"missing.npy": "x"}})

    def test_a_symlink_out_of_the_folder_is_refused(self, tmp_path):
        folder = self.folder(tmp_path)
        outside = tmp_path / "outside.txt"
        outside.write_text("customer data")
        (folder / "link.txt").symlink_to(outside)
        with pytest.raises(ValueError, match="not a file of the build folder"):
            PackageGateway._allowed_files(folder, {"artifacts": {"link.txt": "x"}})


class _Decisions:
    def __len__(self):
        return 0


class TestWaveThatAddsNothing:
    """It still embedded, clustered and called reviewers: the bill is read, not assumed zero."""

    @pytest.fixture
    def package(self, monkeypatch, tmp_path):
        state = {"ledger": 0.37}

        class Engine:
            def __init__(self, cfg, **kw):
                self.cfg = cfg

            def run_wave(self, chunks, **kw):
                ledger = (
                    Path(self.cfg["llm"]["cache_dir"]).parent
                    / "store"
                    / "tx-1"
                    / "_waves"
                    / kw["new_version"]
                )
                ledger.mkdir(parents=True)
                if state["ledger"] is not None:
                    (ledger / "cost_ledger.json").write_text(
                        json.dumps({"total_usd": state["ledger"]})
                    )
                return types.SimpleNamespace(
                    decisions=_Decisions(), new_taxonomy_path=None, summary={}
                )

        artifacts = types.ModuleType("taxonomy_classifier.taxonomy.artifacts")
        artifacts.load = lambda path: types.SimpleNamespace(taxonomy_id="tx-1")
        tax_pkg = types.ModuleType("taxonomy_classifier.taxonomy")
        tax_pkg.artifacts = artifacts
        root = types.ModuleType("taxonomy_classifier")
        root.TaxonomyEngine = Engine
        root.LocalTaxonomyStore = lambda path: types.SimpleNamespace(path=path)
        for name, mod in (
            ("taxonomy_classifier", root),
            ("taxonomy_classifier.taxonomy", tax_pkg),
            ("taxonomy_classifier.taxonomy.artifacts", artifacts),
        ):
            monkeypatch.setitem(sys.modules, name, mod)
        return state

    def run(self, tmp_path):
        gateway = PackageGateway({"llm": {}, "embedding": {}})
        serving = taxonomy(NEW)
        return gateway.propose(
            [{"chunk_id": "c", "document_id": "d", "text": "t"}],
            serving=serving,
            candidate_version="acme-next",
            work_dir=tmp_path / "work",
            spend_cap_usd=1.0,
        )

    def test_the_cost_of_a_wave_without_a_new_category_is_its_ledger(self, tmp_path, package):
        out = self.run(tmp_path)
        assert out.cost_usd == 0.37 and out.report["cost_usd"] == 0.37 and out.files == {}

    def test_a_wave_whose_ledger_is_missing_is_an_unknown_bill_not_a_free_one(
        self, tmp_path, package
    ):
        package["ledger"] = None
        with pytest.raises(CostUnknown):
            self.run(tmp_path)


class TestReportCategory:
    """A report category carries the counts Meridian requires, whoever created the category."""

    def entry(self, **category):
        return PackageGateway._category_entry({"category_id": "CAT_NEW_001", **category}, {})

    def test_a_build_category_reports_its_own_counts(self):
        out = self.entry(member_count=12, document_count=4, stats={"cat_documents": 9})
        assert (out["member_count"], out["document_count"]) == (12, 4)

    def test_a_wave_category_without_a_document_count_reports_the_one_in_its_stats(self):
        out = self.entry(member_count=12, stats={"cat_members": 12, "cat_documents": 3})
        assert out["document_count"] == 3

    def test_a_count_of_zero_is_kept_not_replaced(self):
        assert self.entry(document_count=0, stats={"cat_documents": 3})["document_count"] == 0


def test_a_gateway_without_a_build_config_cannot_discover():
    from airweave.domains.taxonomy.jobs.gateway import PackageGateway

    assert PackageGateway({"llm": {}}).can_discover
    assert not PackageGateway(None).can_discover
