from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

app = FastAPI(title="RAG Retrieval API")


# Build DATABASE_URL from environment if not explicitly provided
def _build_database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    host = (
        os.environ.get("POSTGRES_SERVER")
        or os.environ.get("POSTGRES_HOST")
        or "localhost"
    )
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = _build_database_url()
engine = create_engine(DATABASE_URL) if DATABASE_URL else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello from FastAPI"}


@app.get("/db")
def db_check() -> dict[str, int] | dict[str, str]:
    if engine is None:
        return {"error": "DATABASE_URL not configured"}
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar_one()
    return {"result": int(result)}
