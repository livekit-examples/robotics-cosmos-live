from __future__ import annotations

import json
import uuid

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)
from sentence_transformers import SentenceTransformer

from cosmos_core.storage import VectorStore
from cosmos_core.types import Document


class MilvusVectorStore(VectorStore):
    """Milvus-backed vector store with built-in embedding generation."""

    def __init__(
        self,
        uri: str,
        token: str | None = None,
        collection_name: str = "cosmos_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._uri = uri
        self._token = token
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
            connect_kwargs: dict = {"uri": self._uri}
            if self._token:
                connect_kwargs["token"] = self._token
            self._client = MilvusClient(**connect_kwargs)

            if not self._client.has_collection(self._collection_name):
                model = self._ensure_model()
                dim = model.get_sentence_embedding_dimension()

                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                ]
                schema = CollectionSchema(fields=fields)
                index_params = self._client.prepare_index_params()
                index_params.add_index(
                    field_name="embedding",
                    index_type="IVF_FLAT",
                    metric_type="COSINE",
                    params={"nlist": 128},
                )
                self._client.create_collection(
                    collection_name=self._collection_name,
                    schema=schema,
                    index_params=index_params,
                )
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for a list of texts."""
        model = self._ensure_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def insert(self, document: Document) -> None:
        """Embed the document content and store in Milvus."""
        client = self._ensure_client()
        vectors = self._embed([document.content])
        data = {
            "id": str(uuid.uuid4()),
            "content": document.content,
            "metadata": json.dumps(document.metadata),
            "embedding": vectors[0],
        }
        client.insert(collection_name=self._collection_name, data=[data])

    async def query(self, text: str, top_k: int = 5) -> list[Document]:
        """Embed query text, search Milvus, and return matching Documents."""
        client = self._ensure_client()
        vectors = self._embed([text])
        results = client.search(
            collection_name=self._collection_name,
            data=vectors,
            limit=top_k,
            output_fields=["content", "metadata"],
        )
        documents: list[Document] = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                metadata_str = entity.get("metadata", "{}")
                documents.append(
                    Document(
                        content=entity.get("content", ""),
                        metadata=json.loads(metadata_str),
                    )
                )
        return documents

    def close(self) -> None:
        """Release client resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
