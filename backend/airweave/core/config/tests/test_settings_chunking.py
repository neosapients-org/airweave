"""Tests for the chunking, taxonomy and search settings (contract airweave-neo/1, section 9)."""

import pytest
from pydantic import ValidationError

from airweave.core.config.tests.test_settings_validators import _build_settings

# Settings as Neo runs them (contract section 9).
_NEO_ENV: dict[str, str] = {
    "CHUNK_TARGET_TOKENS": "300",
    "CHUNK_OVERLAP_TOKENS": "45",
    "CHUNK_HEADER_ENABLED": "true",
    "CHUNKER_MODE": "sentence_pack",
    "CHUNKER_VERSION": "2",
    "TAXONOMY_ENABLED": "true",
    "TAXONOMY_S3_BUCKET": "acme-taxonomy",
    "SEARCH_FILTER_GROUPS_OR": "true",
}


class TestDefaults:
    """Every default keeps upstream behaviour."""

    def test_chunking_defaults(self, monkeypatch):
        s = _build_settings(monkeypatch, {})
        assert s.CHUNK_TARGET_TOKENS == 4096
        assert s.CHUNK_OVERLAP_TOKENS == 0
        assert s.CHUNK_HEADER_ENABLED is False
        assert s.CHUNKER_MODE == "semantic"
        assert s.CHUNKER_VERSION == "1"

    def test_taxonomy_defaults(self, monkeypatch):
        s = _build_settings(monkeypatch, {})
        assert s.TAXONOMY_ENABLED is False
        assert s.TAXONOMY_S3_BUCKET is None
        assert s.TAXONOMY_ACTIVE_CACHE_SECONDS == 300
        assert s.TAXONOMY_LOCAL_DIR == "/tmp/airweave/taxonomy"
        assert s.TAXONOMY_EMBED_CACHE_DIR == "/tmp/airweave/taxonomy-cache"
        assert s.TAXONOMY_CLASSIFY_THREADS == 2
        assert s.TAXONOMY_CLASSIFY_TIMEOUT_SECONDS == 60

    def test_relabel_defaults_approve_nothing(self, monkeypatch):
        s = _build_settings(monkeypatch, {})
        assert s.RELABEL_SPEND_CAP_USD == 0.0  # no job can start until the owner approves a cap
        assert s.RELABEL_USD_PER_THOUSAND_CHUNKS == 0.006
        assert (s.RELABEL_BATCH_SIZE, s.RELABEL_PAGE_SIZE, s.RELABEL_WRITE_CONCURRENCY) == (
            64,
            100,
            8,
        )
        assert (s.RELABEL_LEASE_SECONDS, s.RELABEL_RESUME_INTERVAL_SECONDS) == (120, 60)

    def test_relabel_cap_is_read(self, monkeypatch):
        assert (
            _build_settings(monkeypatch, {"RELABEL_SPEND_CAP_USD": "12.5"}).RELABEL_SPEND_CAP_USD
            == 12.5
        )

    @pytest.mark.parametrize("name", ["RELABEL_SPEND_CAP_USD", "RELABEL_USD_PER_THOUSAND_CHUNKS"])
    def test_relabel_money_cannot_be_negative(self, monkeypatch, name):
        with pytest.raises(ValidationError, match=name):
            _build_settings(monkeypatch, {name: "-1"})

    @pytest.mark.parametrize(
        "name",
        [
            "RELABEL_BATCH_SIZE",
            "RELABEL_PAGE_SIZE",
            "RELABEL_WRITE_CONCURRENCY",
            "RELABEL_LEASE_SECONDS",
            "RELABEL_RESUME_INTERVAL_SECONDS",
        ],
    )
    def test_relabel_counts_must_be_positive(self, monkeypatch, name):
        with pytest.raises(ValidationError, match=name):
            _build_settings(monkeypatch, {name: "0"})

    def test_search_default(self, monkeypatch):
        s = _build_settings(monkeypatch, {})
        assert s.SEARCH_FILTER_GROUPS_OR is False

    def test_no_taxonomy_version_setting(self, monkeypatch):
        """The version is resolved per sync from the collection (contract C1), never globally."""
        s = _build_settings(monkeypatch, {})
        assert not [name for name in type(s).model_fields if name.startswith("TAXONOMY_VERSION")]


class TestNeoValues:
    def test_neo_values_load(self, monkeypatch):
        s = _build_settings(monkeypatch, _NEO_ENV)
        assert s.CHUNK_TARGET_TOKENS == 300
        assert s.CHUNK_OVERLAP_TOKENS == 45
        assert s.CHUNK_HEADER_ENABLED is True
        assert s.CHUNKER_MODE == "sentence_pack"
        assert s.CHUNKER_VERSION == "2"
        assert s.TAXONOMY_ENABLED is True
        assert s.TAXONOMY_S3_BUCKET == "acme-taxonomy"
        assert s.SEARCH_FILTER_GROUPS_OR is True


class TestValidation:
    def test_unknown_chunker_mode_rejected(self, monkeypatch):
        with pytest.raises(ValidationError, match="CHUNKER_MODE"):
            _build_settings(monkeypatch, {"CHUNKER_MODE": "neural"})

    @pytest.mark.parametrize("target", ["0", "8193"])
    def test_target_out_of_range_rejected(self, monkeypatch, target):
        with pytest.raises(ValidationError, match="CHUNK_TARGET_TOKENS"):
            _build_settings(monkeypatch, {"CHUNK_TARGET_TOKENS": target})

    @pytest.mark.parametrize("overlap", ["-1", "300", "301"])
    def test_overlap_must_be_below_target(self, monkeypatch, overlap):
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
            _build_settings(
                monkeypatch, {"CHUNK_TARGET_TOKENS": "300", "CHUNK_OVERLAP_TOKENS": overlap}
            )

    @pytest.mark.parametrize(
        "name", ["TAXONOMY_CLASSIFY_THREADS", "TAXONOMY_CLASSIFY_TIMEOUT_SECONDS"]
    )
    def test_taxonomy_counts_must_be_positive(self, monkeypatch, name):
        with pytest.raises(ValidationError, match=name):
            _build_settings(monkeypatch, {name: "0"})


class TestMoneyAndLimits:
    """Caps must be finite; limits that mean 'off' at 0 must not be negative."""

    @pytest.mark.parametrize(
        "name",
        ["DISCOVER_SPEND_CAP_USD", "EVOLUTION_SPEND_CAP_USD", "RELABEL_SPEND_CAP_USD"],
    )
    @pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
    def test_a_cap_that_is_not_a_finite_amount_is_refused(self, monkeypatch, name, value):
        with pytest.raises(ValidationError):
            _build_settings(monkeypatch, {name: value})

    @pytest.mark.parametrize(
        "name", ["TAXONOMY_EMBED_CACHE_MAX_MB", "TAXONOMY_JOBS_CACHE_RETENTION_DAYS"]
    )
    def test_a_negative_limit_is_a_configuration_error_not_unlimited(self, monkeypatch, name):
        with pytest.raises(ValidationError):
            _build_settings(monkeypatch, {name: "-1"})
        assert getattr(_build_settings(monkeypatch, {name: "0"}), name) == 0

    @pytest.mark.parametrize("name", ["TAXONOMY_ARTIFACT_MAX_MB", "TAXONOMY_ARTIFACT_TOTAL_MAX_MB"])
    def test_artifact_limits_need_at_least_one_megabyte(self, monkeypatch, name):
        with pytest.raises(ValidationError):
            _build_settings(monkeypatch, {name: "0"})

    def test_tenant_bindings_are_read_from_json(self, monkeypatch):
        s = _build_settings(monkeypatch, {"TAXONOMY_TENANT_BY_ORGANIZATION": '{"org-1": "t-acme"}'})
        assert s.TAXONOMY_TENANT_BY_ORGANIZATION == {"org-1": "t-acme"}

    def test_no_organization_has_a_tenant_by_default(self, monkeypatch):
        assert _build_settings(monkeypatch, {}).TAXONOMY_TENANT_BY_ORGANIZATION == {}


class TestHeaderBudget:
    def test_a_target_plus_the_header_budget_must_fit_the_embedder(self, monkeypatch):
        with pytest.raises(ValidationError, match="8192"):
            _build_settings(
                monkeypatch,
                {
                    "CHUNK_TARGET_TOKENS": "8000",
                    "CHUNK_HEADER_MAX_TOKENS": "256",
                    "CHUNK_HEADER_ENABLED": "true",
                },
            )

    def test_the_same_target_is_fine_without_headers(self, monkeypatch):
        s = _build_settings(
            monkeypatch, {"CHUNK_TARGET_TOKENS": "8000", "CHUNK_HEADER_ENABLED": "false"}
        )
        assert s.CHUNK_TARGET_TOKENS == 8000

    @pytest.mark.parametrize("value", ["0", "-5", "5000"])
    def test_the_budget_is_bounded(self, monkeypatch, value):
        with pytest.raises(ValidationError):
            _build_settings(monkeypatch, {"CHUNK_HEADER_MAX_TOKENS": value})
