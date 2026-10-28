"""The jobs worker's build settings: a profile the classifier package ships, or a YAML file."""

import sys
import types

import pytest

from airweave.domains.taxonomy.jobs.worker import load_build_config


def stub_profiles(monkeypatch, files):
    """A ``taxonomy_classifier.profiles`` that ships ``files`` (name -> path)."""
    profiles = types.ModuleType("taxonomy_classifier.profiles")

    def path(name):
        if name not in files:
            raise ValueError(f"unknown taxonomy profile {name!r}")
        return files[name]

    profiles.path = path
    package = types.ModuleType("taxonomy_classifier")
    package.profiles = profiles
    monkeypatch.setitem(sys.modules, "taxonomy_classifier", package)
    monkeypatch.setitem(sys.modules, "taxonomy_classifier.profiles", profiles)


def test_no_argument_means_no_build_settings():
    assert load_build_config(None) is None
    assert load_build_config("") is None


def test_a_path_is_read_as_the_packages_yaml(tmp_path):
    f = tmp_path / "taxonomy-build.yaml"
    f.write_text("taxonomy_chunking:\n  target_tokens: 300\n")
    assert load_build_config(str(f)) == {"taxonomy_chunking": {"target_tokens": 300}}


def test_a_name_is_a_profile_the_package_ships(tmp_path, monkeypatch):
    f = tmp_path / "small-corpus.yaml"
    f.write_text("taxonomy_chunking:\n  target_tokens: 70\n")
    stub_profiles(monkeypatch, {"small-corpus": f})
    assert load_build_config("small-corpus") == {"taxonomy_chunking": {"target_tokens": 70}}


def test_a_name_the_package_does_not_ship_is_refused(tmp_path, monkeypatch):
    stub_profiles(monkeypatch, {})
    with pytest.raises(ValueError, match="unknown taxonomy profile"):
        load_build_config("large-corpus")


def test_the_installed_packages_small_corpus_profile_loads():
    pytest.importorskip("taxonomy_classifier.profiles")
    cfg = load_build_config("small-corpus")
    assert cfg["taxonomy_chunking"]["target_tokens"] == 70
    assert not any(k.startswith("_") for k in cfg)  # nothing machine-specific reaches a build
