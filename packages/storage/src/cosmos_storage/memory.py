from __future__ import annotations

from cosmos_core import Document, VectorStore


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for development."""

    async def insert(self, document: Document) -> None:
        raise NotImplementedError

    async def query(self, text: str, top_k: int = 5, since: float | None = None, score_threshold: float | None = None) -> list[Document]:
        raise NotImplementedError
