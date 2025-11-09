import pathlib
from typing import Any

from src.chunkers.base import BaseChunker


class NaiveChunker(BaseChunker):
    """
    A simple baseline chunker that splits text into fixed-size word chunks.

    This strategy ignores sentence boundaries and document structure.
    It’s used as a baseline to compare against more advanced chunkers.
    """

    def chunk(self, text: str) -> list[dict[str, Any]]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        total_words = len(words)

        for i in range(0, total_words, step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk = {
                "id": len(chunks),
                "text": chunk_text,
                "meta": {
                    "start_word": i,
                    "end_word": i + len(chunk_words),
                    "total_words": total_words,
                },
            }
            chunks.append(chunk)

        return chunks


if __name__ == "__main__":

    data_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
    sample_path = data_dir / "arxiv_paper.txt"

    if not sample_path.exists():
        print(f"Sample file not found: {sample_path}")
        exit(1)

    text = sample_path.read_text(encoding="utf-8")
    chunker = NaiveChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk(text)

    print(f"Created {len(chunks)} chunks")
    print(f"First chunk (preview):\n\n{chunks[0]['text'][:500]}...")
