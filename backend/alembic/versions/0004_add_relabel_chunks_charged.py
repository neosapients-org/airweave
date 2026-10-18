"""add taxonomy_relabel_job.chunks_charged.

The chunks a job has sent to the classifier, saved before each batch is sent, so the spend cap
counts work whose result was lost with a worker.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-18 10:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    """Add the column, zero for jobs that already ran."""
    op.add_column(
        "taxonomy_relabel_job",
        sa.Column("chunks_charged", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade():
    """Drop the column."""
    op.drop_column("taxonomy_relabel_job", "chunks_charged")
