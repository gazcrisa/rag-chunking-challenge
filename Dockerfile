FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install Poetry
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy dependency files and install runtime deps
COPY ./pyproject.toml ./poetry.lock* /app/
RUN poetry install --no-root --only main

# Pre-download NLTK data and sentence-transformers model
RUN python -c "import nltk; nltk.download('punkt', quiet=True)" && \
    python -c "from sentence_transformers import SentenceTransformer as ST; ST('all-MiniLM-L6-v2')"

# Copy application code and migrations
COPY ./entrypoint.sh /app/entrypoint.sh
COPY ./migrations /app/migrations
COPY ./alembic.ini /app/alembic.ini
COPY ./src /app/src

# Prepare runtime
RUN chmod +x /app/entrypoint.sh
EXPOSE 8000

# Allow overriding the app module path if needed
ENV APP_MODULE=src.main:app

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
