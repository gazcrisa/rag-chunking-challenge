"""
enable pgvector
"""

from alembic import op

# Revision identifiers used by Alembic
revision = "5a0cd9d67410"
down_revision = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
