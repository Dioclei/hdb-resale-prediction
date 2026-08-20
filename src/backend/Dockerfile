FROM python:3.14.6-slim-trixie

# uv managed environment
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Step 1: install just the dependencies (project source not needed yet)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Step 2: copy source, then install the project itself
COPY README.md ./
COPY alembic ./alembic
COPY alembic.ini alembic.ini
COPY src/backend ./src/backend
# Note that trained models in src/backend/model are copied too
RUN uv sync --frozen --no-dev

# Make sure entrypoint.sh can be executed
RUN chmod +x src/backend/entrypoint.sh

EXPOSE 8000
CMD ["src/backend/entrypoint.sh"]