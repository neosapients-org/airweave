"""add temporal_config to collection

Revision ID: 0001
Revises: 0000
Create Date: 2026-05-08 20:30:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0001"
down_revision = "0000"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "collection",
        sa.Column("temporal_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade():
    op.drop_column("collection", "temporal_config")
