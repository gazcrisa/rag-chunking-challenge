from __future__ import annotations

from typing import Any

import nltk

from src.chunkers.base import BaseChunker

# Ensure the Punkt sentence tokenizer is available once at import time
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
# Newer NLTK versions require 'punkt_tab' alongside 'punkt'
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        # Some NLTK variants may not have punkt_tab; ignore if unavailable
        pass


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunker with sliding-window overlap for context preservation.

    - Splits text into sentences (nltk.sent_tokenize)
    - Packs consecutive sentences until the approximate word budget (chunk_size) is reached
    - Preserves the last `overlap` sentences into the next chunk (sliding window)
    """

    def chunk(self, text: str) -> list[dict[str, Any]]:
        # Tokenize text into sentences
        sentences: list[str] = nltk.sent_tokenize(text or "")
        total_sentences = len(sentences)
        if total_sentences == 0:
            return []

        # Prevent pathological overlap (e.g., overlap >= total sentences)
        overlap_cap = max(0, min(self.overlap, total_sentences - 1))

        chunks: list[dict[str, Any]] = []
        i = 0
        chunk_index = 0

        while i < total_sentences:
            start_idx = i
            current_chunk: list[str] = []
            word_count = 0

            # Collect sentences up to approximately chunk_size words
            while i < total_sentences:
                sentence_words = len(sentences[i].split())
                # If adding this sentence exceeds budget and we already have some content, stop
                if word_count + sentence_words > self.chunk_size and current_chunk:
                    break
                current_chunk.append(sentences[i])
                word_count += sentence_words
                i += 1

            # Safety: if no progress was made (e.g., a single very long sentence), force add it
            if not current_chunk and i < total_sentences:
                current_chunk.append(sentences[i])
                i += 1

            if not current_chunk:
                break

            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "id": chunk_index,
                    "text": chunk_text,
                    "meta": {
                        "start_sentence": start_idx,
                        "end_sentence": i - 1,
                        "total_sentences": total_sentences,
                    },
                }
            )
            chunk_index += 1

            # Apply sliding-window overlap: step back by overlap_cap sentences, but always advance by at least 1
            if overlap_cap > 0:
                i = max(start_idx + 1, i - overlap_cap)

        return chunks
