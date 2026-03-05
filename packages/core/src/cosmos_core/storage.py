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
    async def query(
        self,
        text: str,
        top_k: int = 5,
        since: float | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """Return the *top_k* most similar documents to *text*.

        Args:
            since: If provided, only consider documents with timestamp > this value.
            score_threshold: Minimum similarity score (0–1 for cosine).  Results
                below this threshold are discarded.  ``None`` means no filtering.
        """
