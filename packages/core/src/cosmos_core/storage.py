from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cosmos_core.types import Document


class VectorStore(ABC):
    """Embedding-based document storage and retrieval."""

    @abstractmethod
    async def insert(self, document: Document) -> None:
        """Store a document with its embedding."""

    @abstractmethod
    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        """Return the *top_k* most similar documents to *text*."""


class GraphStore(ABC):
    """Relationship-aware knowledge storage and traversal."""

    @abstractmethod
    async def insert(self, node: dict[str, Any], edges: list[dict[str, Any]] | None = None) -> None:
        """Insert a node and optional edges into the graph."""

    @abstractmethod
    async def query(self, query: str) -> list[dict[str, Any]]:
        """Traverse the graph according to *query* and return matching nodes/edges."""
