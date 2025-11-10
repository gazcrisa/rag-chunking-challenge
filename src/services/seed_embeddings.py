import os
from typing import Any

import openai
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.chunkers.fixed import FixedChunker
from src.models import ChunkStrategy, DocumentChunk
from src.settings import get_settings

# Load settings
settings = get_settings()
openai.api_key = settings.openai_api_key
model = settings.openai_embeddings_model
engine = create_engine(settings.build_sync_sqlalchemy_url())
SessionLocal = sessionmaker(bind=engine)


def seed_embeddings():
    session = SessionLocal()
    data_dir = "data/processed"
    chunker = FixedChunker(chunk_size=500)

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks: list[dict[str, Any]] = chunker.chunk(text)
        print(f"Processing {filename} ({len(chunks)} chunks)...")

        for i, chunk in enumerate(chunks):
            embedding = (
                openai.embeddings.create(model=model, input=chunk["text"])
                .data[0]
                .embedding
            )

            db_obj = DocumentChunk(
                document_name=filename,
                chunk_index=i,
                chunk_text=chunk["text"],
                embedding=embedding,
                strategy=ChunkStrategy.FIXED,
            )
            session.add(db_obj)

        session.commit()
        print(f"{filename} inserted into DB.")

    session.close()


if __name__ == "__main__":
    seed_embeddings()
