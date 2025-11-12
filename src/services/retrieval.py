import openai
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models import ChunkStrategy
from src.settings import get_settings

settings = get_settings()
openai.api_key = settings.openai_api_key
EMBEDDING_MODEL = settings.openai_embeddings_model


class RetrievalService:
    def __init__(self, db: Session):
        self.db = db

    def search(
        self, query: str, top_k: int = 3, strategy: ChunkStrategy = ChunkStrategy.FIXED
    ):
        # Step 1: Create embedding for query
        query_embedding = (
            openai.embeddings.create(model=EMBEDDING_MODEL, input=query)
            .data[0]
            .embedding
        )

        strategy_value = strategy.value

        # Step 2: Build vector similarity query
        sql = """
        SELECT
            document_name,
            chunk_index,
            chunk_text,
            strategy,
            embedding <-> CAST(:query_embedding AS vector) AS distance
        FROM document_chunks
        WHERE strategy = CAST(:strategy AS chunk_strategy)
        ORDER BY embedding <-> CAST(:query_embedding AS vector)
        LIMIT :top_k;
        """

        rows = (
            self.db.execute(
                text(sql),
                {
                    "query_embedding": query_embedding,
                    "strategy": strategy_value,
                    "top_k": top_k,
                },
            )
            .mappings()
            .all()
        )

        # Convert RowMapping objects to plain dicts for JSON serialization
        return [dict(r) for r in rows]
