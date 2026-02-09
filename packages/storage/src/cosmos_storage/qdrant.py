from __future__ import annotations

import logging
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from cosmos_core.storage import VectorStore
from cosmos_core.types import Document

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant vector store with built-in embedding generation.

    Connects to a Qdrant server at *url* (e.g. ``http://localhost:6333``).
    Unlike Milvus, ``search()`` uses native ``order_by`` on the timestamp
    payload field so ``top_k`` limiting happens *after* sorting.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "cosmos_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._model: SentenceTransformer | None = None
        self._client: QdrantClient | None = None

    def _ensure_model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self._embedding_model_name)
        return self._model

    def _ensure_client(self) -> QdrantClient:
        """Lazy-create the QdrantClient and auto-create collection if missing."""
        if self._client is None:
            self._client = QdrantClient(url=self._url)

            if not self._client.collection_exists(self._collection_name):
                model = self._ensure_model()
                dim = model.get_sentence_embedding_dimension()
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                # Payload index on timestamp is required for order_by in scroll().
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="timestamp",
                    field_schema=PayloadSchemaType.FLOAT,
                )
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for a list of texts."""
        model = self._ensure_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    async def insert(self, document: Document) -> None:
        """Embed the document content and upsert into Qdrant."""
        client = self._ensure_client()
        vectors = self._embed([document.content])
        point = PointStruct(
            id=uuid4().hex,
            vector=vectors[0],
            payload={
                "content": document.content,
                "track_id": document.track_id,
                "kind": document.kind,
                "timestamp": document.timestamp,
            },
        )
        logger.info(
            "Inserting document into Qdrant:\n"
            "  participant: %s\n"
            "  kind:        %s\n"
            "  timestamp:   %.2f\n"
            "  content:     %s",
            document.track_id,
            document.kind,
            document.timestamp,
            document.content[:300],
        )
        client.upsert(collection_name=self._collection_name, points=[point])

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
        Qdrant's ``scroll`` with ``order_by`` applies the sort *before*
        the limit, so ``top_k=1`` genuinely returns the newest document.
        """
        client = self._ensure_client()

        must_conditions: list[FieldCondition] = []
        if kind is not None:
            must_conditions.append(FieldCondition(key="kind", match=MatchValue(value=kind)))
        if track_id is not None:
            must_conditions.append(FieldCondition(key="track_id", match=MatchValue(value=track_id)))
        if since is not None:
            must_conditions.append(FieldCondition(key="timestamp", range=Range(gt=since)))

        scroll_filter = Filter(must=must_conditions) if must_conditions else None

        records, _ = client.scroll(
            collection_name=self._collection_name,
            scroll_filter=scroll_filter,
            limit=top_k,
            with_payload=True,
            order_by=OrderBy(key="timestamp", direction="desc"),
        )

        return [
            Document(
                content=r.payload.get("content", ""),
                track_id=r.payload.get("track_id", ""),
                kind=r.payload.get("kind", ""),
                timestamp=r.payload.get("timestamp", 0.0),
            )
            for r in records
        ]

    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        """Embed query text, search Qdrant, and return matching Documents."""
        client = self._ensure_client()
        vectors = self._embed([text])
        results = client.query_points(
            collection_name=self._collection_name,
            query=vectors[0],
            limit=top_k,
            with_payload=True,
        )

        return [
            Document(
                content=point.payload.get("content", ""),
                track_id=point.payload.get("track_id", ""),
                kind=point.payload.get("kind", ""),
                timestamp=point.payload.get("timestamp", 0.0),
            )
            for point in results.points
        ]

    def close(self) -> None:
        """Release client resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
