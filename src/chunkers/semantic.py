from __future__ import annotations

from typing import Any

import nltk
import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunkers.base import BaseChunker

# Ensure NLTK punkt tokenizer is available
nltk.download("punkt", quiet=True)


class SemanticChunker(BaseChunker):
    """
    Embedding-aware chunker that groups adjacent sentences based on semantic similarity.

    A new chunk is created when:
      • cosine similarity between consecutive sentences < similarity_threshold, or
      • adding the next sentence exceeds the chunk_size (word budget).

    Returns:
        A list of chunk dictionaries, each containing id, text, and metadata about
        sentence indices, average similarity, and total sentence count.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        similarity_threshold: float = 0.75,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(chunk_size=chunk_size)
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)

    def chunk(self, text: str) -> list[dict[str, Any]]:
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # Generate embeddings for all sentences at once
        embeddings = self.model.encode(sentences, convert_to_numpy=True, batch_size=32)
        total_sentences = len(sentences)

        chunks: list[dict[str, Any]] = []
        current_chunk: list[str] = []
        similarities_in_chunk: list[float] = []
        chunk_start_idx = 0
        word_count = 0

        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())

            # Check if adding this sentence would exceed the word budget
            if current_chunk and word_count + sentence_words > self.chunk_size:
                self._finalize_chunk(
                    chunks,
                    current_chunk,
                    chunk_start_idx,
                    i - 1,
                    similarities_in_chunk,
                    total_sentences,
                )
                current_chunk, similarities_in_chunk = [], []
                word_count = 0
                chunk_start_idx = i

            # Compute similarity to previous sentence if applicable
            if current_chunk and i > 0:
                sim = float(
                    cosine_similarity(embeddings[i - 1 : i], embeddings[i : i + 1])[0][
                        0
                    ]
                )
                if sim < self.similarity_threshold:
                    # Finalize current chunk due to semantic break
                    self._finalize_chunk(
                        chunks,
                        current_chunk,
                        chunk_start_idx,
                        i - 1,
                        similarities_in_chunk,
                        total_sentences,
                    )
                    current_chunk, similarities_in_chunk = [], []
                    word_count = 0
                    chunk_start_idx = i
                else:
                    similarities_in_chunk.append(sim)

            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            word_count += sentence_words

        # Finalize the last chunk
        if current_chunk:
            self._finalize_chunk(
                chunks,
                current_chunk,
                chunk_start_idx,
                total_sentences - 1,
                similarities_in_chunk,
                total_sentences,
            )

        return chunks

    def _finalize_chunk(
        self,
        chunks: list[dict[str, Any]],
        current_chunk: list[str],
        start_idx: int,
        end_idx: int,
        similarities: list[float],
        total_sentences: int,
    ):
        """
        Helper method to finalize and append a chunk to the chunks list.

        If no similarities were recorded (first sentence or single-sentence chunk),
        defaults avg_similarity to 1.0.
        """
        avg_sim = float(np.mean(similarities)) if similarities else 1.0
        chunks.append(
            {
                "id": len(chunks),
                "text": " ".join(current_chunk),
                "meta": {
                    "start_sentence": start_idx,
                    "end_sentence": end_idx,
                    "avg_similarity": round(avg_sim, 3),
                    "total_sentences": total_sentences,
                },
            }
        )
