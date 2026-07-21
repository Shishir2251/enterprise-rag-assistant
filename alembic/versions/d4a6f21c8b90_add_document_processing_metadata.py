"""add document processing metadata

Revision ID: d4a6f21c8b90
Revises: b7d3c9e21f6a
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "d4a6f21c8b90"
down_revision: str | None = "b7d3c9e21f6a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "progress",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.add_column(
        "documents",
        sa.Column("current_step", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column(
            "processing_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "processing_completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "documents",
        sa.Column("task_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column(
            "retry_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )

    op.execute(
        """
        UPDATE documents
        SET
            status = 'ready',
            progress = 100,
            current_step = 'completed',
            processing_completed_at = updated_at
        WHERE status = 'completed'
        """
    )
    op.execute(
        """
        UPDATE documents
        SET current_step = 'failed'
        WHERE status = 'failed'
        """
    )


def downgrade() -> None:
    op.execute(
        "UPDATE documents SET status = 'completed' WHERE status = 'ready'"
    )
    op.drop_column("documents", "retry_count")
    op.drop_column("documents", "task_id")
    op.drop_column("documents", "processing_completed_at")
    op.drop_column("documents", "processing_started_at")
    op.drop_column("documents", "current_step")
    op.drop_column("documents", "progress")
