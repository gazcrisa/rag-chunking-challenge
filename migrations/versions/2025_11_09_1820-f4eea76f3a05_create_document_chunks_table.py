"""
create document chunks table
"""

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# Revision identifiers used by Alembic
revision = "f4eea76f3a05"
down_revision = "5a0cd9d67410"


def upgrade():

    # Create enum for strategy
    chunk_strategy = ["FIXED", "SENTENCE", "SEMANTIC", "SLIDING_WINDOW"]
    sa.Enum(*chunk_strategy, name="chunk_strategy").create(op.get_bind())

    # Create document_chunks table
    op.create_table(
        "document_chunks",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("document_name", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column(
            "strategy",
            pg.ENUM(*chunk_strategy, name="chunk_strategy", create_type=False),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade():
    op.drop_table("document_chunks")
    op.execute("DROP TYPE IF EXISTS chunk_strategy")
