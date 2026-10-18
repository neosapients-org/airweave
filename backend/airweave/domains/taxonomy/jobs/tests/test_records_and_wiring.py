"""The job table, the Temporal registrations and the container wiring of the taxonomy jobs."""

import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from airweave.core.container.factory import _create_taxonomy_job_service
from airweave.domains.taxonomy.jobs import records as r
from airweave.domains.taxonomy.jobs.tests.helpers import job
from airweave.models.taxonomy_job import TaxonomyJob

NOW = datetime(2026, 10, 15, 9, 0, 0)


def sql(statement) -> str:
    return " ".join(str(statement.compile(dialect=postgresql.dialect())).split())


class TestStatements:
    def test_claim_needs_an_active_job_and_a_free_lease(self):
        text = sql(r.claim_statement(uuid4(), "w1", 300, NOW))
        assert (
            "state IN" in text and "lease_owner IS NULL" in text and "lease_expires_at <=" in text
        )

    def test_renew_and_finish_are_fenced_by_the_lease_owner(self):
        assert "lease_owner =" in sql(r.renew_statement(uuid4(), "w1", 300, NOW, {"phase": "x"}))
        assert "lease_owner =" in sql(r.finish_statement(uuid4(), "w1", "FAILED", "e", NOW, {}))

    def test_cancel_stops_at_once_only_without_a_live_lease(self):
        flag, stop = r.cancel_statements(uuid4(), NOW)
        assert "cancel_requested" in sql(flag) and "lease_expires_at <=" in sql(stop)


class TestModel:
    def test_one_active_job_per_collection_and_a_key_per_collection(self):
        idx = next(
            i
            for i in TaxonomyJob.__table__.indexes
            if i.name == "uq_taxonomy_job_active_collection"
        )
        assert idx.unique and "PENDING" in str(idx.dialect_options["postgresql"]["where"])
        keys = {
            c.name: [col.name for col in c.columns]
            for c in TaxonomyJob.__table__.constraints
            if c.name
        }
        assert keys["uq_taxonomy_job_key"] == ["collection_id", "idempotency_key"]

    def test_round_trip_and_columns_match_the_record(self):
        j = job(params={"sample_size": 5}, result={"x": 1})
        assert r.to_record(r.to_row(j)).params == {"sample_size": 5}
        columns = {c.name for c in TaxonomyJob.__table__.columns} - {"modified_at"}
        assert columns == set(r.TaxonomyJobRecord.__dataclass_fields__)

    def test_migration_follows_0002_and_creates_every_column(self):
        path = Path(__file__).resolve().parents[5] / "alembic/versions/0003_add_taxonomy_job.py"
        spec = importlib.util.spec_from_file_location("m0003", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert (m.revision, m.down_revision) == ("0003", "0002")
        source = path.read_text()
        for column in TaxonomyJob.__table__.columns:
            assert f'"{column.name}"' in source, column.name


class TestRegistrations:
    def test_the_sync_worker_registers_no_taxonomy_job_activity_or_workflow(self):
        from airweave.domains.taxonomy.jobs.temporal import ACTIVITY_NAME, WORKFLOW_NAME
        from airweave.domains.temporal.worker import wiring

        names = [w.__name__ for w in wiring.get_workflows()]
        assert "TaxonomyJobWorkflow" not in names and WORKFLOW_NAME not in names
        source = Path(wiring.__file__).read_text()
        for needle in ("taxonomy.jobs", "TaxonomyJob", ACTIVITY_NAME, "taxonomy_job"):
            assert needle not in source

    def test_the_sync_worker_module_never_imports_the_jobs_package(self):
        import airweave.domains.temporal.worker as sync_worker

        assert "taxonomy.jobs" not in Path(sync_worker.__file__).read_text()

    def test_the_jobs_worker_registers_only_the_job_workflow_and_activity(self):
        from airweave.domains.taxonomy.jobs.temporal import ACTIVITY_NAME
        from airweave.domains.taxonomy.jobs.worker import registrations

        workflows, activities = registrations(MagicMock())
        assert [w.__name__ for w in workflows] == ["TaxonomyJobWorkflow"]
        assert len(activities) == 1
        assert getattr(activities[0], "__temporal_activity_definition").name == ACTIVITY_NAME

    def test_the_jobs_task_queue_is_not_the_sync_queue(self):
        from airweave.core.config import settings
        from airweave.domains.taxonomy import limits

        assert limits.JOBS_TASK_QUEUE != settings.TEMPORAL_TASK_QUEUE

    async def test_the_activity_runs_the_job_and_returns_its_state(self):
        from uuid import UUID

        from airweave.domains.taxonomy.jobs.temporal import TaxonomyJobActivity

        runner = SimpleNamespace(run_job=AsyncMock(return_value="COMPLETED"))
        activity = TaxonomyJobActivity(runner)
        assert await activity.run(str(UUID(int=5))) == "COMPLETED"
        runner.run_job.assert_awaited_once_with(UUID(int=5))

    async def test_start_workflow_uses_the_job_id_so_a_replay_is_a_no_op(self):
        from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy

        from airweave.domains.taxonomy.jobs.temporal import start_job_workflow

        client = MagicMock()
        client.start_workflow = AsyncMock()
        jid = uuid4()
        await start_job_workflow(client, "q-jobs", jid)
        args, kwargs = client.start_workflow.await_args
        assert kwargs["id"] == f"taxonomy-job-{jid}" and kwargs["task_queue"] == "q-jobs"
        # A running workflow is the same job (no second one); an ended one is started again, which
        # is how the reconciler recovers a job whose row is still active.
        assert kwargs["id_conflict_policy"] == WorkflowIDConflictPolicy.USE_EXISTING
        assert kwargs["id_reuse_policy"] == WorkflowIDReusePolicy.ALLOW_DUPLICATE


class TestContainer:
    def settings(self, **kw):
        base = {
            "TAXONOMY_JOBS_ENABLED": True,
            "TAXONOMY_SPEND_CAP_USD": 2.0,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def test_off_builds_nothing(self):
        assert (
            _create_taxonomy_job_service(self.settings(TAXONOMY_JOBS_ENABLED=False), object())
            is None
        )
        assert _create_taxonomy_job_service(self.settings(), None) is None

    def test_on_builds_the_service_with_the_one_ceiling(self):
        svc = _create_taxonomy_job_service(self.settings(), object())
        assert svc._cap == 2.0


class TestSettings:
    def test_defaults_approve_nothing_and_are_off(self, monkeypatch):
        from airweave.core.config.tests.test_settings_chunking import _build_settings

        s = _build_settings(monkeypatch, {})
        assert s.TAXONOMY_JOBS_ENABLED is False
        assert s.TAXONOMY_SPEND_CAP_USD == 0.0

    def test_the_jobs_numbers_are_constants_not_settings(self):
        from airweave.domains.taxonomy import limits

        assert limits.JOBS_TASK_QUEUE == "airweave-taxonomy-jobs-queue"
        assert (limits.DISCOVER_SAMPLE_DOCUMENTS, limits.DISCOVER_MIN_DOCUMENTS) == (200, 1)


class TestReturningReadsBeforeCommit:
    """A commit expires the row; reading it afterwards fails outside the greenlet on a real database."""

    async def test_the_row_is_read_before_the_commit(self):
        committed = {"yes": False}
        values = {c: None for c in r._COLUMNS}
        values.update(params={}, id=uuid4())

        class ExpiringRow:
            def __getattr__(self, name):
                if committed["yes"]:
                    raise RuntimeError("MissingGreenlet: the row expired at commit")
                return values[name]

        class Db:
            async def execute(self, _statement):
                return SimpleNamespace(scalar_one_or_none=lambda: ExpiringRow())

            async def commit(self):
                committed["yes"] = True

        class Session:
            async def __aenter__(self):
                return Db()

            async def __aexit__(self, *_):
                return False

        repo = r.SqlTaxonomyJobRepository(session=Session)
        record = await repo._returning(object())  # the statement is not inspected by the fake

        assert record is not None and record.id == values["id"]


class TestWorkflowWaitsForAHeldLease:
    """The first attempt of a crashed worker holds the lease; the second sees "busy". The workflow
    waits for the lease to expire and asks again, instead of ending with the job still RUNNING.
    """

    async def test_it_asks_again_after_busy_and_stops_at_the_first_real_state(self):
        from airweave.domains.taxonomy.jobs import temporal as t

        answers, slept = iter(["busy", "busy", "COMPLETED"]), []

        async def run():
            return next(answers)

        async def sleep(duration):
            slept.append(duration)

        assert await t.run_until_not_busy(run, sleep) == "COMPLETED"
        assert slept == [t.BUSY_WAIT, t.BUSY_WAIT]

    async def test_it_gives_up_with_busy_when_a_live_worker_keeps_the_lease(self):
        from airweave.domains.taxonomy.jobs import temporal as t

        calls = []

        async def run():
            calls.append(1)
            return "busy"

        async def sleep(_):
            return None

        assert await t.run_until_not_busy(run, sleep, attempts=3) == "busy"
        assert len(calls) == 3


class TestPackageErrorsAreTranslated:
    def test_corpus_too_small_becomes_too_little_text(self):
        from airweave.domains.taxonomy.jobs.gateway import translate_package_error
        from airweave.domains.taxonomy.jobs.sampling import TooFewDocuments

        class CorpusTooSmall(
            ValueError
        ):  # the package's class, by name: the package is not imported here
            pass

        mapped = translate_package_error(
            CorpusTooSmall("no category has the 4 members needed to calibrate")
        )
        assert isinstance(mapped, TooFewDocuments)
        assert "not enough text yet" in str(mapped) and "Add more documents" in str(mapped)

    def test_any_other_error_is_left_alone(self):
        from airweave.domains.taxonomy.jobs.gateway import translate_package_error

        error = ValueError("something else")
        assert translate_package_error(error) is error
