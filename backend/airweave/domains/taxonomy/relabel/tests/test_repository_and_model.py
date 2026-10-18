"""The job table: model, migration and the SQL of each state change.

No database is available in this batch, so the statements are compiled for PostgreSQL and read;
the same rules are exercised on ``InMemoryJobRepository`` in ``test_runner.py``.
"""

import importlib.util
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from airweave.domains.taxonomy.relabel import repository as r
from airweave.domains.taxonomy.relabel.jobs import RelabelJob
from airweave.domains.taxonomy.relabel.tests.helpers import new_job
from airweave.models.taxonomy_relabel_job import TaxonomyRelabelJob

NOW = datetime(2026, 10, 15, 9, 0, 0)


def sql(statement) -> str:
    return " ".join(str(statement.compile(dialect=postgresql.dialect())).split())


class TestStatements:
    def test_claim_only_when_active_and_the_lease_is_free_mine_or_expired(self):
        text = sql(r.claim_statement(uuid4(), "w1", 120, NOW))
        assert "state IN" in text and "lease_owner IS NULL" in text
        assert "lease_owner =" in text and "lease_expires_at <=" in text
        assert "RETURNING" in text

    def test_progress_is_fenced_by_the_lease_owner(self):
        text = sql(
            r.progress_statement(uuid4(), "w1", 120, NOW, {"checkpoint": "x", "chunks_failed": 1})
        )
        assert "lease_owner =" in text and "state IN" in text and "checkpoint" in text

    def test_finish_frees_the_lease_and_is_fenced(self):
        text = sql(r.finish_statement(uuid4(), "w1", "COMPLETED", None, NOW, {}))
        assert "lease_owner =" in text and "finished_at" in text
        assert "lease_owner=NULL" in text.replace(
            " ", ""
        ) or "lease_owner=%(lease_owner)s" in text.replace(" ", "")

    def test_cancel_stops_at_once_only_without_a_live_lease(self):
        flag, stop = r.cancel_statements(uuid4(), NOW)
        assert "cancel_requested" in sql(flag)
        text = sql(stop)
        assert "lease_expires_at IS NULL" in text and "lease_expires_at <=" in text

    def test_resumable_is_active_with_a_free_lease_oldest_first(self):
        text = sql(r.resumable_query(NOW))
        assert "state IN" in text and "ORDER BY" in text and "lease_expires_at <=" in text

    def test_unannounced_is_ended_and_not_sent(self):
        text = sql(r.unannounced_query())
        assert "completion_event_sent IS false" in text.lower().replace("is false", "IS false")


class TestModelAndMapping:
    def test_one_active_job_per_collection_is_a_unique_partial_index(self):
        index = next(
            i
            for i in TaxonomyRelabelJob.__table__.indexes
            if i.name == "uq_relabel_job_active_collection"
        )
        assert index.unique and [c.name for c in index.columns] == ["collection_id"]
        assert "PENDING" in str(index.dialect_options["postgresql"]["where"])

    def test_the_idempotency_key_is_unique_per_collection(self):
        names = {
            c.name: [col.name for col in c.columns]
            for c in TaxonomyRelabelJob.__table__.constraints
            if c.name
        }
        assert names["uq_relabel_job_key"] == ["collection_id", "idempotency_key"]

    def test_row_and_dataclass_round_trip(self):
        job = new_job(idempotency_key="k", checkpoint="[1, null]", chunks_relabelled=4)
        row = r.to_row(job)
        assert isinstance(row, TaxonomyRelabelJob)
        back = r.to_job(row)
        assert isinstance(back, RelabelJob) and back.id == job.id
        assert (back.idempotency_key, back.checkpoint, back.chunks_relabelled) == (
            "k",
            "[1, null]",
            4,
        )

    def test_every_column_has_a_field_in_the_dataclass(self):
        columns = {c.name for c in TaxonomyRelabelJob.__table__.columns} - {"modified_at"}
        assert columns == set(RelabelJob.__dataclass_fields__)


class TestMigration:
    def load(self):
        path = (
            Path(__file__).resolve().parents[5]
            / "alembic/versions/0002_add_taxonomy_relabel_job.py"
        )
        spec = importlib.util.spec_from_file_location("m0002", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_follows_the_previous_head_and_has_both_directions(self):
        m = self.load()
        assert (m.revision, m.down_revision) == ("0002", "0001")
        assert callable(m.upgrade) and callable(m.downgrade)

    def test_creates_every_model_column(self):
        versions = Path(__file__).resolve().parents[5] / "alembic/versions"
        # the table (0002) and the columns added to it later (0004)
        source = (versions / "0002_add_taxonomy_relabel_job.py").read_text() + (
            versions / "0004_add_relabel_chunks_charged.py"
        ).read_text()
        for column in TaxonomyRelabelJob.__table__.columns:
            assert f'"{column.name}"' in source, column.name
        assert "uq_relabel_job_active_collection" in source and "uq_relabel_job_key" in source


class TestReturningReadsBeforeCommit:
    """A commit expires the row; reading it afterwards fails outside the greenlet on a real database."""

    async def test_the_row_is_read_before_the_commit(self):
        from types import SimpleNamespace
        from uuid import uuid4

        from airweave.domains.taxonomy.relabel import repository as repo_mod

        committed = {"yes": False}
        values = {c: None for c in repo_mod._COLUMNS}
        values["id"] = uuid4()

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

        repo = repo_mod.SqlJobRepository(session=Session)
        job = await repo._returning(object())

        assert job is not None and job.id == values["id"]
