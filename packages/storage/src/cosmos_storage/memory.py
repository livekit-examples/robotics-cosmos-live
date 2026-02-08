from __future__ import annotations

from typing import Any

from cosmos_core import Document, VectorStore, GraphStore


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for development."""

    async def insert(self, document: Document) -> None:
        raise NotImplementedError

    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        raise NotImplementedError


class InMemoryGraphStore(GraphStore):
    """Simple in-memory graph store for development."""

    async def insert(self, node: dict[str, Any], edges: list[dict[str, Any]] | None = None) -> None:
        raise NotImplementedError

    async def query(self, query: str) -> list[dict[str, Any]]:
        raise NotImplementedError
