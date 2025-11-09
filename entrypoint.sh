#!/usr/bin/env bash
set -euo pipefail

# If there are no migration files, stamp base; otherwise upgrade head
if [ -d "/app/migrations/versions" ] && ls -1 /app/migrations/versions/*.py >/dev/null 2>&1; then
  alembic upgrade head
else
  alembic stamp base || true
fi

# Start server
APP_MODULE=${APP_MODULE:-src.main:app}
exec uvicorn "$APP_MODULE" --host 0.0.0.0 --port 8000 --reload