import uuid
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, Column
from sqlalchemy import Enum as SAEnum
from sqlalchemy import Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ChunkStrategy(StrEnum):
    FIXED = "FIXED"
    SENTENCE = "SENTENCE"
    SEMANTIC = "SEMANTIC"
    SLIDING_WINDOW = "SLIDING_WINDOW"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_name = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    strategy = Column(SAEnum(ChunkStrategy, name="chunk_strategy"), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
