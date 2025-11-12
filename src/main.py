from __future__ import annotations

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from src.database import get_db
from src.models import ChunkStrategy
from src.services.retrieval import RetrievalService

app = FastAPI(title="RAG Retrieval API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(..., description="User query"),
    strategy: ChunkStrategy = Query(default=ChunkStrategy.FIXED),
    top_k: int = Query(3),
    db=Depends(get_db),
):
    results = RetrievalService(db).search(q, top_k=top_k, strategy=strategy)
    return {"query": q, "results": results}
