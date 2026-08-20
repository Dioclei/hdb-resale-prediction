#!/bin/sh
set -e

echo "Running database migrations..."
uv run --no-dev alembic upgrade head

echo "Starting FastAPI application..."
exec uv run --no-dev uvicorn src.backend.main:app --host 0.0.0.0 --port 8000