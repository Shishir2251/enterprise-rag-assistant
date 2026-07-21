"""use local embedding dimension and provider metadata

Revision ID: e8f3a1b6c2d4
Revises: d4a6f21c8b90

Existing vectors cannot be resized safely. This migration deliberately clears
only embedding values and their metadata; documents and chunk content remain
unchanged and can be reindexed with the active provider.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e8f3a1b6c2d4"
down_revision: str | None = "d4a6f21c8b90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding_provider",
            sa.String(length=50),
            nullable=True,
        ),
    )
    op.execute(
        """
        UPDATE document_chunks
        SET
            embedding = NULL,
            embedding_model = NULL,
            embedding_provider = NULL,
            embedded_at = NULL
        """
    )
    op.execute(
        """
        ALTER TABLE document_chunks
        ALTER COLUMN embedding
        TYPE extensions.vector(384)
        USING NULL::extensions.vector(384)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE document_chunks
        SET
            embedding = NULL,
            embedding_model = NULL,
            embedding_provider = NULL,
            embedded_at = NULL
        """
    )
    op.execute(
        """
        ALTER TABLE document_chunks
        ALTER COLUMN embedding
        TYPE extensions.vector(1536)
        USING NULL::extensions.vector(1536)
        """
    )
    op.drop_column("document_chunks", "embedding_provider")
