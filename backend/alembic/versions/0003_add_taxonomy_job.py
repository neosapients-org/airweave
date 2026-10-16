"""add taxonomy_job

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-15 15:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "taxonomy_job",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("modified_at", sa.DateTime(), nullable=False),
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("collection_readable_id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("candidate_version", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("phase", sa.String(), nullable=True),
        sa.Column("spend_cap_usd", sa.Float(), nullable=False),
        sa.Column("spend_used_usd", sa.Float(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("lease_owner", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["collection_id"], ["collection.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "idempotency_key", name="uq_taxonomy_job_key"),
    )
    op.create_index(
        op.f("ix_taxonomy_job_organization_id"), "taxonomy_job", ["organization_id"]
    )
    op.create_index("ix_taxonomy_job_state", "taxonomy_job", ["state"])
    op.create_index(
        "uq_taxonomy_job_active_collection",
        "taxonomy_job",
        ["collection_id"],
        unique=True,
        postgresql_where=sa.text("state IN ('PENDING', 'RUNNING')"),
    )


def downgrade():
    op.drop_index("uq_taxonomy_job_active_collection", table_name="taxonomy_job")
    op.drop_index("ix_taxonomy_job_state", table_name="taxonomy_job")
    op.drop_index(op.f("ix_taxonomy_job_organization_id"), table_name="taxonomy_job")
    op.drop_table("taxonomy_job")
