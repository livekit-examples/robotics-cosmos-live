from __future__ import annotations

import logging

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

from cosmos_core.storage import VectorStore
from cosmos_core.types import Document


class MilvusVectorStore(VectorStore):
    """Milvus Lite vector store with built-in embedding generation.

    Uses a local ``.db`` file via Milvus Lite — no server required.
    """

    def __init__(
        self,
        db_path: str = "cosmos.db",
        collection_name: str = "cosmos_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._db_path = db_path
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._model: SentenceTransformer | None = None
        self._client: MilvusClient | None = None

    def _ensure_model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self._embedding_model_name)
        return self._model

    def _ensure_client(self) -> MilvusClient:
        """Lazy-create the MilvusClient and auto-create collection if missing."""
        if self._client is None:
            self._client = MilvusClient(self._db_path)

            if not self._client.has_collection(self._collection_name):
                model = self._ensure_model()
                dim = model.get_sentence_embedding_dimension()
                self._client.create_collection(
                    collection_name=self._collection_name,
                    dimension=dim,
                    auto_id=True,
                )
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for a list of texts."""
        model = self._ensure_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    async def insert(self, document: Document) -> None:
        """Embed the document content and store in Milvus Lite."""
        client = self._ensure_client()
        vectors = self._embed([document.content])
        data = {
            "content": document.content,
            "track_id": document.track_id,
            "kind": document.kind,
            "timestamp": document.timestamp,
            "vector": vectors[0],
        }
        logger.info(
            "Inserting document into Milvus:\n"
            "  participant: %s\n"
            "  kind:        %s\n"
            "  timestamp:   %.2f\n"
            "  content:     %s",
            document.track_id,
            document.kind,
            document.timestamp,
            document.content[:300],
        )
        client.insert(collection_name=self._collection_name, data=[data])

    async def search(
        self,
        expr: str = "",
        top_k: int = 10,
    ) -> list[Document]:
        """Filter-based search. No embeddings — like a SQL WHERE clause.

        Fields: ``track_id`` (str), ``kind`` (str), ``timestamp`` (float), ``content`` (str).
        Results are ordered by timestamp descending (most recent first).

        Examples::

            'track_id == "cam-1"'
            'timestamp > 1700000000'
            'track_id == "cam-1" and timestamp > 1700000000'
            'kind == "analysis"'
            'track_id in ["cam-1", "cam-2"]'
        """
        client = self._ensure_client()

        results = client.query(
            collection_name=self._collection_name,
            filter=expr or "id >= 0",
            output_fields=["content", "track_id", "kind", "timestamp"],
            limit=top_k,
        )
        results.sort(key=lambda r: r.get("timestamp", 0.0), reverse=True)
        return [
            Document(
                content=r.get("content", ""),
                track_id=r.get("track_id", ""),
                kind=r.get("kind", ""),
                timestamp=r.get("timestamp", 0.0),
            )
            for r in results
        ]

    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        """Embed query text, search Milvus Lite, and return matching Documents."""
        client = self._ensure_client()
        vectors = self._embed([text])
        results = client.search(
            collection_name=self._collection_name,
            data=vectors,
            limit=top_k,
            output_fields=["content", "track_id", "kind", "timestamp"],
        )
        documents: list[Document] = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                documents.append(
                    Document(
                        content=entity.get("content", ""),
                        track_id=entity.get("track_id", ""),
                        kind=entity.get("kind", ""),
                        timestamp=entity.get("timestamp", 0.0),
                    )
                )
        return documents

    def close(self) -> None:
        """Release client resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
