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
        assert s.TAXONOMY_S3_PREFIX == "taxonomy/"
        assert s.TAXONOMY_ACTIVE_CACHE_SECONDS == 300
        assert s.TAXONOMY_LOCAL_DIR == "/tmp/airweave/taxonomy"
        assert s.TAXONOMY_EMBED_CACHE_DIR == "/tmp/airweave/taxonomy-cache"
        assert s.TAXONOMY_CLASSIFY_THREADS == 2
        assert s.TAXONOMY_CLASSIFY_TIMEOUT_SECONDS == 60

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
