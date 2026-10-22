"""The classifier package is checked when a process starts, against the interface Airweave calls."""

import inspect

import pytest

from airweave.domains.taxonomy.package import (
    JOBS_INTERFACE,
    SYNC_INTERFACE,
    ClassifierPackageMissing,
    require_classifier_package,
)
from airweave.domains.taxonomy.tests.package_stub import stub_classifier_package


class TestTheCheck:
    def test_passes_when_every_module_and_name_is_there(self, monkeypatch):
        stub_classifier_package(monkeypatch)
        require_classifier_package(SYNC_INTERFACE, JOBS_INTERFACE)

    def test_names_a_missing_module(self, monkeypatch):
        stub_classifier_package(monkeypatch, without=("taxonomy_classifier.contract",))
        with pytest.raises(ClassifierPackageMissing, match="taxonomy_classifier.contract"):
            require_classifier_package()

    def test_names_a_missing_name(self, monkeypatch):
        stub_classifier_package(
            monkeypatch, without=("taxonomy_classifier.integration.airweave.TaxonomyClassifier",)
        )
        with pytest.raises(ClassifierPackageMissing, match="TaxonomyClassifier"):
            require_classifier_package()

    def test_the_sync_check_does_not_need_the_build_extra(self, monkeypatch):
        stub_classifier_package(monkeypatch, without=("taxonomy_classifier.taxonomy.artifacts",))
        require_classifier_package(SYNC_INTERFACE)
        with pytest.raises(ClassifierPackageMissing, match="artifacts"):
            require_classifier_package(SYNC_INTERFACE, JOBS_INTERFACE)


class TestTheContainerFailsAtBoot:
    def test_taxonomy_on_without_the_package_refuses_to_build_the_container(self, monkeypatch):
        from types import SimpleNamespace

        from airweave.core.container.factory import _create_taxonomy_processor

        stub_classifier_package(monkeypatch, without=("taxonomy_classifier.contract",))
        settings = SimpleNamespace(
            TAXONOMY_ENABLED=True,
            TAXONOMY_S3_BUCKET="",
            TAXONOMY_DATA_DIR="/tmp/none",
            TAXONOMY_CLASSIFY_THREADS=1,
        )
        with pytest.raises(ClassifierPackageMissing):
            _create_taxonomy_processor(settings)


class TestTheRealPackage:
    """Run where the package is installed (the image): its interface is the one Airweave calls."""

    @pytest.fixture(autouse=True)
    def _installed(self):
        pytest.importorskip("taxonomy_classifier")

    def test_it_provides_every_name(self):
        require_classifier_package(SYNC_INTERFACE, JOBS_INTERFACE)

    def test_the_classifier_and_the_contract_take_what_airweave_passes(self):
        from taxonomy_classifier.contract import to_contract
        from taxonomy_classifier.integration.airweave import TaxonomyClassifier

        # ClassifierPool calls TaxonomyClassifier(path, config, transport) and
        # to_contract(enriched_row, classifier.taxonomy).
        inspect.signature(TaxonomyClassifier).bind("path", {}, None)
        inspect.signature(to_contract).bind({}, None)
        assert callable(getattr(TaxonomyClassifier, "classify", None))
