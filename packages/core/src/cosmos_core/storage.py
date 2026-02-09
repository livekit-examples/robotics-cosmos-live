from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_core.types import Document


class VectorStore(ABC):
    """Embedding-based document storage and retrieval."""

    @abstractmethod
    async def insert(self, document: Document) -> None:
        """Store a document with its embedding."""

    @abstractmethod
    async def search(
        self,
        *,
        kind: str | None = None,
        track_id: str | None = None,
        since: float | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """Filter-based search on document metadata.

        Results are ordered by timestamp descending (most recent first).
        """

    @abstractmethod
    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        """Return the *top_k* most similar documents to *text*."""
