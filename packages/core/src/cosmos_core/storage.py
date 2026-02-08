from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_core.types import Document


class VectorStore(ABC):
    """Embedding-based document storage and retrieval."""

    @abstractmethod
    async def insert(self, document: Document) -> None:
        """Store a document with its embedding."""

    @abstractmethod
    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        """Return the *top_k* most similar documents to *text*."""
