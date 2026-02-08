from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cosmos_core.types import Document
from cosmos_live.config import MilvusConfig
from cosmos_storage.milvus import MilvusVectorStore


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_milvus_config_defaults():
    cfg = MilvusConfig(uri="http://localhost:19530")
    assert cfg.collection_name == "cosmos_documents"
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.token is None


def test_milvus_config_custom():
    cfg = MilvusConfig(
        uri="http://milvus:19530",
        token="secret",
        collection_name="my_docs",
        embedding_model="custom-model",
    )
    assert cfg.collection_name == "my_docs"
    assert cfg.embedding_model == "custom-model"
    assert cfg.token.get_secret_value() == "secret"


# ---------------------------------------------------------------------------
# Constructor — no eager connections
# ---------------------------------------------------------------------------


def test_constructor_no_eager_connections():
    store = MilvusVectorStore(uri="http://localhost:19530")
    assert store._model is None
    assert store._client is None


# ---------------------------------------------------------------------------
# _embed
# ---------------------------------------------------------------------------


@patch("cosmos_storage.milvus.SentenceTransformer", autospec=True)
def test_embed_calls_model_encode(mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_st_cls.return_value = mock_model

    store = MilvusVectorStore(uri="http://localhost:19530")
    result = store._embed(["hello world"])

    mock_st_cls.assert_called_once_with("all-MiniLM-L6-v2")
    mock_model.encode.assert_called_once_with(["hello world"], normalize_embeddings=True)
    assert result == [[0.1, 0.2, 0.3]]


# ---------------------------------------------------------------------------
# _ensure_model — lazy loading
# ---------------------------------------------------------------------------


@patch("cosmos_storage.milvus.SentenceTransformer", autospec=True)
def test_ensure_model_lazy_loads_once(mock_st_cls):
    mock_model = MagicMock()
    mock_st_cls.return_value = mock_model

    store = MilvusVectorStore(uri="http://localhost:19530")
    m1 = store._ensure_model()
    m2 = store._ensure_model()

    assert m1 is m2
    mock_st_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _ensure_client — collection auto-creation
# ---------------------------------------------------------------------------


@patch("cosmos_storage.milvus.SentenceTransformer", autospec=True)
def test_ensure_client_creates_collection_when_missing(mock_st_cls):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st_cls.return_value = mock_model

    with patch("cosmos_storage.milvus.MilvusClient", autospec=True) as mock_client_cls, \
         patch("cosmos_storage.milvus.CollectionSchema") as mock_schema_cls, \
         patch("cosmos_storage.milvus.FieldSchema"), \
         patch("cosmos_storage.milvus.DataType") as mock_dtype:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client

        store = MilvusVectorStore(uri="http://localhost:19530")
        client = store._ensure_client()

        mock_client.has_collection.assert_called_once_with("cosmos_documents")
        mock_client.create_collection.assert_called_once()
        assert client is mock_client


@patch("cosmos_storage.milvus.SentenceTransformer", autospec=True)
def test_ensure_client_skips_creation_when_exists(mock_st_cls):
    with patch("cosmos_storage.milvus.MilvusClient", autospec=True) as mock_client_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client_cls.return_value = mock_client

        store = MilvusVectorStore(uri="http://localhost:19530")
        client = store._ensure_client()

        mock_client.create_collection.assert_not_called()
        assert client is mock_client


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_generates_embedding_and_stores():
    store = MilvusVectorStore(uri="http://localhost:19530")

    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    store._client = mock_client

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7]])
    store._model = mock_model

    doc = Document(content="test content", metadata={"key": "value"})
    await store.insert(doc)

    mock_model.encode.assert_called_once_with(["test content"], normalize_embeddings=True)
    mock_client.insert.assert_called_once()
    call_kwargs = mock_client.insert.call_args
    data = call_kwargs.kwargs["data"][0]
    assert data["content"] == "test content"
    assert json.loads(data["metadata"]) == {"key": "value"}
    assert data["embedding"] == [0.5, 0.6, 0.7]
    assert len(data["id"]) == 36  # UUID length


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_returns_documents():
    store = MilvusVectorStore(uri="http://localhost:19530")

    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = [
        [
            {"entity": {"content": "doc1", "metadata": '{"k": "v1"}'}},
            {"entity": {"content": "doc2", "metadata": '{"k": "v2"}'}},
        ]
    ]
    store._client = mock_client

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    store._model = mock_model

    results = await store.query("search text", top_k=2)

    mock_model.encode.assert_called_once_with(["search text"], normalize_embeddings=True)
    mock_client.search.assert_called_once()
    assert len(results) == 2
    assert results[0].content == "doc1"
    assert results[0].metadata == {"k": "v1"}
    assert results[1].content == "doc2"
    assert results[1].metadata == {"k": "v2"}


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_releases_client():
    store = MilvusVectorStore(uri="http://localhost:19530")
    mock_client = MagicMock()
    store._client = mock_client

    store.close()

    mock_client.close.assert_called_once()
    assert store._client is None


def test_close_safe_when_no_client():
    store = MilvusVectorStore(uri="http://localhost:19530")
    store.close()  # should not raise
    assert store._client is None
