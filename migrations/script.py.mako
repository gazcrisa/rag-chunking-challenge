"""
${message}
"""

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# Revision identifiers used by Alembic
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}


def upgrade():
    ${upgrades if upgrades else "pass"}
