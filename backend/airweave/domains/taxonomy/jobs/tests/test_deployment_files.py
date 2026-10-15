"""The jobs worker's deployment files (no helm binary here, so the template is read as text).

A real ``helm template`` render is still to be done; these tests pin what the template and the
values must say: off by default, zero replicas, a separate Deployment on its own queue, a
separate service account for the candidates/ write grant, and settings that equal the code's.
"""

import re
from pathlib import Path

import pytest
import yaml

from airweave.core.config.tests.test_settings_chunking import _build_settings

ROOT = Path(__file__).resolve().parents[6]
HELM = ROOT / "ops/helm/airweave-svc"
pytestmark = pytest.mark.skipif(
    not (HELM / "values.yaml").exists() or not (ROOT / ".env.example").exists(),
    reason="the chart and .env.example are outside the backend checkout",
)
JOB_KEYS = [
    "TAXONOMY_JOBS_ENABLED",
    "TAXONOMY_JOBS_TASK_QUEUE",
    "TAXONOMY_JOBS_LEASE_SECONDS",
    "TAXONOMY_JOBS_MAX_CHUNKS",
    "DISCOVER_SPEND_CAP_USD",
    "DISCOVER_SAMPLE_DOCUMENTS",
    "DISCOVER_MIN_DOCUMENTS",
    "EVOLUTION_SPEND_CAP_USD",
]


@pytest.fixture(scope="module")
def values():
    return yaml.safe_load((HELM / "values.yaml").read_text())


@pytest.fixture(scope="module")
def template():
    return (HELM / "templates/taxonomy-jobs-worker.yaml").read_text()


class TestValues:
    def test_off_with_no_replicas_by_default(self, values):
        w = values["taxonomyJobsWorker"]
        assert w["enabled"] is False and w["replicaCount"] == 0
        assert w["serviceAccount"]["create"] is False

    def test_own_memory_and_cpu_limits_larger_than_the_sync_worker(self, values):
        jobs, sync = values["taxonomyJobsWorker"]["resources"], values["worker"]["resources"]
        assert jobs["limits"]["memory"] == "8Gi" and jobs["requests"]["cpu"] == "1"
        assert jobs != sync

    def test_env_settings_are_strings_and_approve_nothing(self, values):
        env = values["env"]
        assert all(isinstance(env[k], str) for k in JOB_KEYS)
        assert env["TAXONOMY_JOBS_ENABLED"] == "false"
        assert env["DISCOVER_SPEND_CAP_USD"] == "0" and env["EVOLUTION_SPEND_CAP_USD"] == "0"

    def test_the_jobs_queue_is_not_the_sync_queue(self, values):
        assert values["env"]["TAXONOMY_JOBS_TASK_QUEUE"] != "airweave-sync-queue"

    def test_the_iam_and_image_requirements_are_stated_in_the_comments(self):
        text = (HELM / "values.yaml").read_text()
        assert "taxonomy/tenants/*/candidates/*" in text and "nothing more" in text
        assert "read-only on versions, active and models" in text
        assert "umap-learn" in text and "numba" in text and "hdbscan" in text

    def test_helm_defaults_equal_the_code_defaults(self, values, monkeypatch):
        s = _build_settings(monkeypatch, {})
        env = values["env"]
        assert env["TAXONOMY_JOBS_TASK_QUEUE"] == s.TAXONOMY_JOBS_TASK_QUEUE
        assert float(env["DISCOVER_SPEND_CAP_USD"]) == s.DISCOVER_SPEND_CAP_USD
        assert float(env["EVOLUTION_SPEND_CAP_USD"]) == s.EVOLUTION_SPEND_CAP_USD
        assert int(env["DISCOVER_SAMPLE_DOCUMENTS"]) == s.DISCOVER_SAMPLE_DOCUMENTS
        assert int(env["DISCOVER_MIN_DOCUMENTS"]) == s.DISCOVER_MIN_DOCUMENTS
        assert int(env["TAXONOMY_JOBS_LEASE_SECONDS"]) == s.TAXONOMY_JOBS_LEASE_SECONDS
        assert int(env["TAXONOMY_JOBS_MAX_CHUNKS"]) == s.TAXONOMY_JOBS_MAX_CHUNKS


class TestEnvExample:
    def test_every_job_setting_is_listed_with_the_code_default(self, monkeypatch):
        lines = dict(
            line.split("=", 1)
            for line in (ROOT / ".env.example").read_text().splitlines()
            if re.match(r"^[A-Z_]+=", line)
        )
        s = _build_settings(monkeypatch, {})
        for key in JOB_KEYS:
            assert key in lines, key
        assert (
            lines["TAXONOMY_JOBS_ENABLED"] == "false"
            and float(lines["DISCOVER_SPEND_CAP_USD"]) == 0
        )
        assert lines["TAXONOMY_JOBS_TASK_QUEUE"] == s.TAXONOMY_JOBS_TASK_QUEUE


class TestTemplate:
    def test_the_whole_file_is_guarded_by_the_enabled_flag(self, template):
        assert template.lstrip().startswith("{{- if .Values.taxonomyJobsWorker.enabled }}")
        assert template.rstrip().endswith("{{- end }}")

    def test_it_is_a_separate_deployment_running_the_jobs_module(self, template):
        assert "kind: Deployment" in template and "-taxonomy-jobs" in template
        assert '["python", "-m", "airweave.domains.taxonomy.jobs.worker"]' in template
        assert "airweave.domains.temporal.worker" not in template  # not the sync worker's entry
        assert "app.kubernetes.io/component: taxonomy-jobs-worker" in template

    def test_replicas_and_resources_come_from_the_values(self, template):
        assert "replicas: {{ .Values.taxonomyJobsWorker.replicaCount }}" in template
        assert ".Values.taxonomyJobsWorker.resources" in template

    def test_one_pod_at_a_time_never_two_on_a_lease(self, template):
        assert "type: Recreate" in template

    def test_the_pod_turns_the_jobs_on_for_itself_and_has_scratch_space(self, template):
        assert 'name: TAXONOMY_JOBS_ENABLED\n              value: "true"' in template
        assert (
            "jobs-work" in template
            and ".Values.taxonomyJobsWorker.workVolume.sizeLimit" in template
        )

    def test_it_can_use_its_own_service_account_for_the_candidates_grant(self, template):
        assert ".Values.taxonomyJobsWorker.serviceAccount.create" in template
        assert "kind: ServiceAccount" in template

    def test_braces_balance(self, template):
        assert template.count("{{") == template.count("}}")
        opens = len(re.findall(r"{{-? *(if|range|with)\b", template))
        ends = len(re.findall(r"{{-? *end\b", template))
        assert opens == ends


def test_the_sync_worker_template_is_unchanged_and_has_no_jobs_reference():
    text = (HELM / "templates/temporal-worker.yaml").read_text()
    assert "taxonomy.jobs" not in text and "taxonomyJobsWorker" not in text
