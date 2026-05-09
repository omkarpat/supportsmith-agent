"""Initialize pgvector support schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260507_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial persistence and vector-search tables."""
    op.execute("create extension if not exists vector")

    op.create_table(
        "support_documents",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("external_id", sa.Text(), nullable=False, unique=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text()),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("category", sa.Text()),
        sa.Column("quality", sa.Text(), nullable=False, server_default="trusted"),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.execute("alter table support_documents add column embedding vector(1536)")
    op.create_index("support_documents_source_idx", "support_documents", ["source"])
    op.create_index("support_documents_category_idx", "support_documents", ["category"])
    op.create_index(
        "support_documents_metadata_gin_idx",
        "support_documents",
        ["metadata"],
        postgresql_using="gin",
    )
    op.execute(
        """
        create index support_documents_embedding_hnsw_idx
            on support_documents using hnsw (embedding vector_cosine_ops)
            where embedding is not null
        """
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("summary", sa.Text()),
    )

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        # Captures the LangSmith root run UUID for the agent / compliance
        # message in this turn. Null on the inbound user row and when
        # tracing is disabled. Stored as a column (not just metadata) so
        # operators can jump from a DB row directly to a LangSmith trace
        # without searching JSON.
        sa.Column("langsmith_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "role in ('user', 'agent', 'compliance')",
            name="conversation_messages_role_check",
        ),
    )
    op.create_index(
        "conversation_messages_conversation_idx",
        "conversation_messages",
        ["conversation_id", "created_at"],
    )
    op.create_index(
        "conversation_messages_conversation_turn_idx",
        "conversation_messages",
        ["conversation_id", "turn_number"],
    )

    op.create_table(
        "escalations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
        ),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "transcript",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    """Drop initial persistence and vector-search tables."""
    op.drop_table("escalations")
    op.drop_index(
        "conversation_messages_conversation_turn_idx",
        table_name="conversation_messages",
    )
    op.drop_index("conversation_messages_conversation_idx", table_name="conversation_messages")
    op.drop_table("conversation_messages")
    op.drop_table("conversations")
    op.execute("drop index if exists support_documents_embedding_hnsw_idx")
    op.drop_index("support_documents_metadata_gin_idx", table_name="support_documents")
    op.drop_index("support_documents_category_idx", table_name="support_documents")
    op.drop_index("support_documents_source_idx", table_name="support_documents")
    op.drop_table("support_documents")
