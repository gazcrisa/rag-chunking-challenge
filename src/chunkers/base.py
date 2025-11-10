from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):
    """
    Abstract base class for all document chunkers.

    Each subclass must implement `chunk()` which takes a raw text
    string and returns a list of chunk objects or plain text strings.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        """
        Args:
            chunk_size: Approximate size of each chunk (in words or tokens).
            overlap: Number of words/tokens that overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, text: str) -> list[dict[str, Any]]:
        """
        Split text into chunks.

        Returns:
            A list of dicts, each containing:
              - 'id': numeric index of chunk
              - 'text': the chunk content
              - 'meta': optional metadata (page, section, etc.)
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, overlap={self.overlap})"
