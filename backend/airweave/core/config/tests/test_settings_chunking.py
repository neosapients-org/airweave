"""Tests for the chunking, taxonomy and search settings (contract airweave-neo/1, section 9).

Only what a deployment decides is a setting. The numbers nobody sets per deployment are constants
(``core/config/chunking.py`` and ``domains/taxonomy/limits.py``); their tests live beside them.
"""

import pytest
from pydantic import ValidationError

from airweave.core.config.chunking import ChunkPlan
from airweave.core.config.settings import Settings
from airweave.core.config.tests.test_settings_validators import _build_settings

# Settings as Neo runs them (contract section 9).
_NEO_ENV: dict[str, str] = {
    "CHUNKER_MODE": "sentence_pack",
    "CHUNK_TARGET_TOKENS": "300",
    "TAXONOMY_ENABLED": "true",
    "TAXONOMY_S3_BUCKET": "acme-taxonomy",
    "SEARCH_FILTER_GROUPS_OR": "true",
}

# Names that used to be settings. They must not come back as settings: each is a constant now.
_RETIRED = (
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_HEADER_ENABLED",
    "CHUNK_HEADER_MAX_TOKENS",
    "CHUNKER_VERSION",
    "TAXONOMY_ACTIVE_CACHE_SECONDS",
    "TAXONOMY_LOCAL_DIR",
    "TAXONOMY_EMBED_CACHE_DIR",
    "TAXONOMY_EMBED_CACHE_MAX_MB",
    "TAXONOMY_ARTIFACT_MAX_MB",
    "TAXONOMY_ARTIFACT_TOTAL_MAX_MB",
    "TAXONOMY_CLASSIFY_TIMEOUT_SECONDS",
    "RELABEL_SPEND_CAP_USD",
    "RELABEL_USD_PER_THOUSAND_CHUNKS",
    "RELABEL_BATCH_SIZE",
    "RELABEL_PAGE_SIZE",
    "RELABEL_WRITE_CONCURRENCY",
    "RELABEL_LEASE_SECONDS",
    "RELABEL_RESUME_INTERVAL_SECONDS",
    "DISCOVER_SPEND_CAP_USD",
    "EVOLUTION_SPEND_CAP_USD",
    "DISCOVER_SAMPLE_DOCUMENTS",
    "DISCOVER_MIN_DOCUMENTS",
    "TAXONOMY_JOBS_TASK_QUEUE",
    "TAXONOMY_JOBS_WORK_DIR",
    "TAXONOMY_JOBS_BUILD_CONFIG",
    "TAXONOMY_JOBS_LEASE_SECONDS",
    "TAXONOMY_JOBS_MAX_CHUNKS",
    "TAXONOMY_JOBS_CACHE_RETENTION_DAYS",
    "TAXONOMY_JOBS_RECONCILE_SECONDS",
    "TAXONOMY_JOBS_DISPATCH_GRACE_SECONDS",
    "TAXONOMY_JOBS_MAX_AGE_HOURS",
)


class TestDefaults:
    """Every default keeps upstream behaviour, and nothing paid can start."""

    def test_chunking_defaults(self, monkeypatch):
        s = _build_settings(monkeypatch, {})
        assert s.CHUNK_TARGET_TOKENS == 4096
        assert s.CHUNKER_MODE == "semantic"

    def test_taxonomy_defaults(self, monkeypatch):
        s = _build_settings(monkeypatch, {})
        assert s.TAXONOMY_ENABLED is False
        assert s.TAXONOMY_S3_BUCKET is None
        assert s.TAXONOMY_DATA_DIR == "/tmp/airweave/taxonomy"
        assert s.TAXONOMY_CLASSIFY_THREADS == 2
        assert s.TAXONOMY_JOBS_ENABLED is False
        assert s.DISCOVER_MIN_CHUNKS == 100
        assert s.TAXONOMY_JOBS_CACHE_DIR == ""

    def test_the_spend_ceiling_approves_nothing_until_the_owner_sets_it(self, monkeypatch):
        assert _build_settings(monkeypatch, {}).TAXONOMY_SPEND_CAP_USD == 0.0

    def test_the_spend_ceiling_is_read(self, monkeypatch):
        s = _build_settings(monkeypatch, {"TAXONOMY_SPEND_CAP_USD": "12.5"})
        assert s.TAXONOMY_SPEND_CAP_USD == 12.5

    def test_search_default(self, monkeypatch):
        assert _build_settings(monkeypatch, {}).SEARCH_FILTER_GROUPS_OR is False

    def test_no_taxonomy_version_setting(self, monkeypatch):
        """The version is resolved per sync from the collection (contract C1), never globally."""
        s = _build_settings(monkeypatch, {})
        assert not [name for name in type(s).model_fields if name.startswith("TAXONOMY_VERSION")]

    def test_the_new_settings_are_the_short_list_a_deployment_decides(self):
        assert not [name for name in _RETIRED if name in Settings.model_fields]


class TestNeoValues:
    def test_neo_values_load(self, monkeypatch):
        s = _build_settings(monkeypatch, _NEO_ENV)
        assert s.CHUNK_TARGET_TOKENS == 300
        assert s.CHUNKER_MODE == "sentence_pack"
        assert s.TAXONOMY_ENABLED is True
        assert s.TAXONOMY_S3_BUCKET == "acme-taxonomy"
        assert s.SEARCH_FILTER_GROUPS_OR is True

    def test_the_neo_plan_follows_from_the_mode(self, monkeypatch):
        s = _build_settings(monkeypatch, _NEO_ENV)
        plan = ChunkPlan.from_settings(s)
        assert (plan.target_tokens, plan.overlap_tokens, plan.header) == (300, 45, True)
        assert plan.positions is True
        assert plan.is_upstream is False

    def test_the_default_plan_is_upstreams(self, monkeypatch):
        plan = ChunkPlan.from_settings(_build_settings(monkeypatch, {}))
        assert (plan.overlap_tokens, plan.header, plan.positions) == (0, False, False)
        assert plan.is_upstream is True


class TestValidation:
    def test_unknown_chunker_mode_rejected(self, monkeypatch):
        with pytest.raises(ValidationError, match="CHUNKER_MODE"):
            _build_settings(monkeypatch, {"CHUNKER_MODE": "neural"})

    @pytest.mark.parametrize("target", ["0", "8193"])
    def test_target_out_of_range_rejected(self, monkeypatch, target):
        with pytest.raises(ValidationError, match="CHUNK_TARGET_TOKENS"):
            _build_settings(monkeypatch, {"CHUNK_TARGET_TOKENS": target})

    def test_a_sentence_pack_target_plus_the_header_must_fit_the_embedder(self, monkeypatch):
        with pytest.raises(ValidationError, match="8192"):
            _build_settings(
                monkeypatch, {"CHUNKER_MODE": "sentence_pack", "CHUNK_TARGET_TOKENS": "8000"}
            )

    def test_the_same_target_is_fine_without_headers(self, monkeypatch):
        s = _build_settings(monkeypatch, {"CHUNK_TARGET_TOKENS": "8000"})
        assert s.CHUNK_TARGET_TOKENS == 8000

    def test_any_valid_target_gives_an_overlap_below_it(self):
        for target in (1, 2, 40, 300, 8000):
            plan = ChunkPlan.for_mode("sentence_pack", target)
            assert 0 <= plan.overlap_tokens < target

    @pytest.mark.parametrize("name", ["TAXONOMY_CLASSIFY_THREADS", "DISCOVER_MIN_CHUNKS"])
    def test_counts_must_be_positive(self, monkeypatch, name):
        with pytest.raises(ValidationError, match=name):
            _build_settings(monkeypatch, {name: "0"})


class TestMoney:
    """The ceiling must be a finite amount from 0 up: NaN compares false with everything."""

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
    def test_a_ceiling_that_is_not_a_finite_amount_is_refused(self, monkeypatch, value):
        with pytest.raises(ValidationError, match="TAXONOMY_SPEND_CAP_USD"):
            _build_settings(monkeypatch, {"TAXONOMY_SPEND_CAP_USD": value})
