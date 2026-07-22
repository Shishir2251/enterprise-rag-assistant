"""extend chat message lifecycle metadata

Revision ID: f9a2c4e7d1b3
Revises: e8f3a1b6c2d4
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "f9a2c4e7d1b3"
down_revision: str | None = "e8f3a1b6c2d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("status", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "chat_messages",
        sa.Column("llm_provider", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "chat_messages",
        sa.Column("llm_model", sa.String(length=100), nullable=True),
    )

    # All pre-Phase-9 rows represent messages that completed persistence.
    op.execute(
        sa.text(
            "UPDATE chat_messages SET status = 'completed' "
            "WHERE status IS NULL"
        )
    )
    op.alter_column(
        "chat_messages",
        "status",
        existing_type=sa.String(length=20),
        nullable=False,
    )
    op.create_check_constraint(
        "ck_chat_messages_status",
        "chat_messages",
        "status IN ('pending', 'completed', 'failed', 'fallback')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_chat_messages_status",
        "chat_messages",
        type_="check",
    )
    op.drop_column("chat_messages", "llm_model")
    op.drop_column("chat_messages", "llm_provider")
    op.drop_column("chat_messages", "status")
