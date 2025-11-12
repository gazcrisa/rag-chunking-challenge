from __future__ import annotations

import os
from itertools import islice
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

from src.chunkers.fixed import FixedChunker
from src.chunkers.semantic import SemanticChunker
from src.chunkers.sentence import SentenceChunker
from src.database import get_db
from src.models import ChunkStrategy, DocumentChunk
from src.settings import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
model = settings.openai_embeddings_model
DATA_DIR = Path("data/processed")


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def batched(iterable, n: int):
    """Yield successive n-sized batches from an iterable."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI embeddings."""
    if not texts:
        return []
    all_embeddings: list[list[float]] = []
    for batch in batched(texts, batch_size):
        resp = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
    return all_embeddings


# --------------------------------------------------------------------
# Main seeding logic
# --------------------------------------------------------------------


def seed_embeddings():
    """Read .txt files, chunk them, embed chunks, and persist to DB."""
    strategy_env = os.getenv("STRATEGY", "FIXED").upper()
    chosen_strategy = ChunkStrategy.__members__.get(strategy_env, ChunkStrategy.FIXED)
    if chosen_strategy == ChunkStrategy.SENTENCE:
        # Sentence-based chunker with sliding window
        logger.info("Using SentenceChunker with sliding window")
        chunker = SentenceChunker(chunk_size=500, overlap=2)
    elif chosen_strategy == ChunkStrategy.SEMANTIC:
        # Semantic, embedding-aware chunker (uses SentenceTransformer internally to form chunks)
        logger.info("Using SemanticChunker with embedding-aware chunking")
        chunker = SemanticChunker(chunk_size=500, similarity_threshold=0.45)
    else:
        logger.info("Using FixedChunker with fixed-size chunking")
        chunker = FixedChunker(chunk_size=500)

    with get_db() as session:
        for path in DATA_DIR.glob("*.txt"):
            logger.info(f"Processing {path.name}...")

            text = path.read_text(encoding="utf-8")
            chunks: list[dict[str, Any]] = chunker.chunk(text)
            logger.info(
                f" - {len(chunks)} chunks generated using {chosen_strategy.value} strategy"
            )

            # Batch embeddings for efficiency
            embeddings = embed_texts([c["text"] for c in chunks])

            db_objects = [
                DocumentChunk(
                    document_name=path.name,
                    chunk_index=i,
                    chunk_text=chunk["text"],
                    embedding=embedding,
                    strategy=chosen_strategy,
                )
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            session.add_all(db_objects)
            logger.info(f" - Inserted {len(db_objects)} chunks into DB")

        session.commit()
        logger.info(
            f"All documents processed successfully with strategy={chosen_strategy.value}"
        )


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    seed_embeddings()
