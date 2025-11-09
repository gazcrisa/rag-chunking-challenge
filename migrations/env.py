import os
from logging.config import fileConfig
from urllib.parse import quote

from alembic import context
from loguru import logger
from sqlalchemy import engine_from_config, pool

# Alembic Config
config = context.config

# Configure logging via .ini
fileConfig(config.config_file_name)  # type: ignore

# Configure DB URL from env -> alembic.ini template
section = config.config_ini_section
postgres_user = os.getenv("POSTGRES_SERVICE_USER", "postgres")
postgres_pass = os.getenv("POSTGRES_SERVICE_PASSWORD", "postgres")
postgres_port = os.getenv("POSTGRES_PORT", "5432")
postgres_host = os.getenv("POSTGRES_SERVER", "localhost")
postgres_db = os.getenv("POSTGRES_DB", "postgres")

config.set_section_option(section, "DB_USER", postgres_user)
config.set_section_option(section, "DB_PASS", quote(postgres_pass).replace("%", "%%"))
config.set_section_option(section, "DB_PORT", postgres_port)
config.set_section_option(section, "DB_HOST", postgres_host)
config.set_section_option(section, "DB_DATABASE", postgres_db)

# No autogenerate metadata for now (manual migrations or add later)
target_metadata = None


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),  # type: ignore
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    with connectable.connect() as connection:
        do_run_migrations(connection)


logger.info(f"migrations/env.py: user={postgres_user} db={postgres_db}")
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
