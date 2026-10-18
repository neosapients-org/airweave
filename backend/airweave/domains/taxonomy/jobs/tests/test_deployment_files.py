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
# What a deployment decides about the jobs; every other number is a constant in the code.
JOB_KEYS = [
    "TAXONOMY_JOBS_ENABLED",
    "TAXONOMY_SPEND_CAP_USD",
    "DISCOVER_MIN_CHUNKS",
    "TAXONOMY_TENANT_BY_ORGANIZATION",
    "TAXONOMY_JOBS_CACHE_DIR",
]
# Retired settings: each is a constant now, so none may come back as an environment entry.
RETIRED_KEYS = [
    "TAXONOMY_JOBS_TASK_QUEUE",
    "TAXONOMY_JOBS_WORK_DIR",
    "TAXONOMY_JOBS_BUILD_CONFIG",
    "TAXONOMY_JOBS_LEASE_SECONDS",
    "TAXONOMY_JOBS_MAX_CHUNKS",
    "TAXONOMY_JOBS_CACHE_RETENTION_DAYS",
    "TAXONOMY_JOBS_RECONCILE_SECONDS",
    "TAXONOMY_JOBS_DISPATCH_GRACE_SECONDS",
    "TAXONOMY_JOBS_MAX_AGE_HOURS",
    "DISCOVER_SPEND_CAP_USD",
    "EVOLUTION_SPEND_CAP_USD",
    "RELABEL_SPEND_CAP_USD",
    "DISCOVER_SAMPLE_DOCUMENTS",
    "DISCOVER_MIN_DOCUMENTS",
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_HEADER_ENABLED",
    "CHUNK_HEADER_MAX_TOKENS",
    "CHUNKER_VERSION",
    "TAXONOMY_LOCAL_DIR",
    "TAXONOMY_EMBED_CACHE_DIR",
    "TAXONOMY_ACTIVE_CACHE_SECONDS",
    "TAXONOMY_ARTIFACT_MAX_MB",
    "TAXONOMY_ARTIFACT_TOTAL_MAX_MB",
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

    def test_the_decisions_are_strings_and_approve_nothing(self, values):
        env = values["env"]
        for key in ("TAXONOMY_JOBS_ENABLED", "TAXONOMY_SPEND_CAP_USD"):
            assert isinstance(env[key], str), key
        assert env["TAXONOMY_JOBS_ENABLED"] == "false"
        assert env["TAXONOMY_SPEND_CAP_USD"] == "0"

    def test_no_retired_setting_is_left_in_the_values(self, values):
        assert not [k for k in RETIRED_KEYS if k in values["env"]]

    def test_the_jobs_queue_is_a_constant_that_is_not_the_sync_queue(self):
        from airweave.domains.taxonomy import limits

        assert limits.JOBS_TASK_QUEUE != "airweave-sync-queue"

    def test_the_iam_and_image_requirements_are_stated_in_the_comments(self):
        text = (HELM / "values.yaml").read_text()
        assert "taxonomy/tenants/*/candidates/*" in text and "nothing more" in text
        assert "read-only on versions, active and models" in text
        assert "umap-learn" in text and "numba" in text and "hdbscan" in text

    def test_the_data_folder_is_the_workers_taxonomy_volume(self, values):
        template = (HELM / "templates/temporal-worker.yaml").read_text()
        assert values["env"]["TAXONOMY_DATA_DIR"] == "/data/airweave-taxonomy"
        assert "mountPath: /data/airweave-taxonomy" in template


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
        assert int(lines["DISCOVER_MIN_CHUNKS"]) == s.DISCOVER_MIN_CHUNKS
        assert lines["TAXONOMY_JOBS_CACHE_DIR"] == s.TAXONOMY_JOBS_CACHE_DIR
        assert lines["TAXONOMY_JOBS_ENABLED"] == "false"
        assert float(lines["TAXONOMY_SPEND_CAP_USD"]) == 0
        assert not [k for k in RETIRED_KEYS if k in lines]


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


class TestBuildConfigRollsThePod:
    def test_the_template_hashes_the_build_config_into_the_pod_annotations(self, template):
        assert re.search(
            r"checksum/build-config:\s*\{\{\s*\.Values\.taxonomyJobsWorker\.buildConfig\s*\|\s*sha256sum",
            template,
        )
        # only while a build config exists, so the pod of a deployment without one is unchanged
        block = template.split("checksum/build-config", 1)[0].rsplit("{{-", 1)[-1]
        assert "if .Values.taxonomyJobsWorker.buildConfig" in block

    def test_a_changed_config_changes_the_rendered_checksum(self):
        import shutil
        import subprocess

        if shutil.which("helm") is None:
            pytest.skip("helm is not installed")

        def render(config):
            out = subprocess.run(
                [
                    "helm",
                    "template",
                    "t",
                    str(HELM),
                    "--set",
                    "taxonomyJobsWorker.enabled=true",
                    "--set",
                    f"taxonomyJobsWorker.buildConfig={config}",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            return re.search(r"checksum/build-config: (\w+)", out).group(1)

        assert render("a: 1") != render("a: 2")
        assert render("a: 1") == render("a: 1")

    def test_the_build_config_is_the_workers_argument_not_an_environment_entry(self, template):
        assert "TAXONOMY_JOBS_BUILD_CONFIG" not in template
        assert 'args: ["/etc/airweave/taxonomy-build.yaml"]' in template

    def test_the_scratch_volume_is_mounted_inside_the_taxonomy_volume(self, template):
        assert "TAXONOMY_JOBS_WORK_DIR" not in template
        assert template.index("mountPath: /data/airweave-taxonomy\n") < template.index(
            "mountPath: /data/airweave-taxonomy/jobs"
        )
