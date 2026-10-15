"""add taxonomy_relabel_job

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-15 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "taxonomy_relabel_job",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("modified_at", sa.DateTime(), nullable=False),
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("collection_readable_id", sa.String(), nullable=False),
        sa.Column("target_version", sa.String(), nullable=False),
        sa.Column("previous_version", sa.String(), nullable=True),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=True),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("spend_cap_usd", sa.Float(), nullable=False),
        sa.Column("spend_estimate_usd", sa.Float(), nullable=False),
        sa.Column("chunks_total", sa.Integer(), nullable=False),
        sa.Column("chunks_relabelled", sa.Integer(), nullable=False),
        sa.Column("chunks_stamped", sa.Integer(), nullable=False),
        sa.Column("chunks_skipped_changed", sa.Integer(), nullable=False),
        sa.Column("chunks_unlabellable", sa.Integer(), nullable=False),
        sa.Column("chunks_failed", sa.Integer(), nullable=False),
        sa.Column("checkpoint", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("lease_owner", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("completion_event_sent", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["collection_id"], ["collection.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "idempotency_key", name="uq_relabel_job_key"),
    )
    op.create_index(
        op.f("ix_taxonomy_relabel_job_organization_id"),
        "taxonomy_relabel_job",
        ["organization_id"],
    )
    op.create_index("ix_relabel_job_state", "taxonomy_relabel_job", ["state"])
    op.create_index(
        "uq_relabel_job_active_collection",
        "taxonomy_relabel_job",
        ["collection_id"],
        unique=True,
        postgresql_where=sa.text("state IN ('PENDING', 'RUNNING')"),
    )


def downgrade():
    op.drop_index("uq_relabel_job_active_collection", table_name="taxonomy_relabel_job")
    op.drop_index("ix_relabel_job_state", table_name="taxonomy_relabel_job")
    op.drop_index(
        op.f("ix_taxonomy_relabel_job_organization_id"), table_name="taxonomy_relabel_job"
    )
    op.drop_table("taxonomy_relabel_job")
