from __future__ import annotations

import os
import tempfile

import pytest

from cosmos_core.types import Document
from cosmos_live.config import MilvusConfig
from cosmos_storage.milvus import MilvusVectorStore


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_milvus_config_defaults():
    cfg = MilvusConfig()
    assert cfg.db_path == "cosmos.db"
    assert cfg.collection_name == "cosmos_documents"
    assert cfg.embedding_model == "all-MiniLM-L6-v2"


def test_milvus_config_custom():
    cfg = MilvusConfig(
        db_path="/tmp/custom.db",
        collection_name="my_docs",
        embedding_model="custom-model",
    )
    assert cfg.db_path == "/tmp/custom.db"
    assert cfg.collection_name == "my_docs"
    assert cfg.embedding_model == "custom-model"


# ---------------------------------------------------------------------------
# Fixture — fresh store per test backed by a temp .db file
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    db_file = str(tmp_path / "test.db")
    s = MilvusVectorStore(
        db_path=db_file,
        collection_name="test_collection",
    )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Constructor — no eager connections
# ---------------------------------------------------------------------------


def test_constructor_no_eager_connections():
    store = MilvusVectorStore(db_path="/tmp/noop.db")
    assert store._model is None
    assert store._client is None


# ---------------------------------------------------------------------------
# _embed — real embeddings
# ---------------------------------------------------------------------------


def test_embed_returns_384d_vectors(store: MilvusVectorStore):
    vecs = store._embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384


def test_embed_batch(store: MilvusVectorStore):
    vecs = store._embed(["foo", "bar", "baz"])
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 384


# ---------------------------------------------------------------------------
# insert + query — real round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_and_query_single_document(store: MilvusVectorStore):
    doc = Document(
        content="the quick brown fox jumps over the lazy dog",
        track_id="cam-1",
        kind="analysis",
        timestamp=1.0,
    )
    await store.insert(doc)

    results = await store.query("quick brown fox", top_k=1)
    assert len(results) == 1
    assert results[0].content == doc.content
    assert results[0].track_id == "cam-1"
    assert results[0].kind == "analysis"
    assert results[0].timestamp == 1.0


@pytest.mark.asyncio
async def test_insert_multiple_and_query_returns_most_relevant(store: MilvusVectorStore):
    docs = [
        Document(content="python programming language", track_id="t1", kind="transcript", timestamp=1.0),
        Document(content="the weather is sunny today", track_id="t2", kind="analysis", timestamp=2.0),
        Document(content="machine learning and artificial intelligence", track_id="t3", kind="transcript", timestamp=3.0),
    ]
    for doc in docs:
        await store.insert(doc)

    results = await store.query("deep learning AI", top_k=2)
    assert len(results) == 2
    # The ML/AI doc should be the top result
    assert results[0].content == "machine learning and artificial intelligence"


@pytest.mark.asyncio
async def test_query_empty_collection_returns_empty(store: MilvusVectorStore):
    results = await store.query("anything", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_query_top_k_limits_results(store: MilvusVectorStore):
    for i in range(5):
        await store.insert(Document(content=f"document number {i}", track_id="t", kind="k", timestamp=float(i)))

    results = await store.query("document", top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_releases_client(store: MilvusVectorStore):
    # Force client creation
    store._ensure_client()
    assert store._client is not None

    store.close()
    assert store._client is None


def test_close_safe_when_no_client():
    s = MilvusVectorStore(db_path="/tmp/noop.db")
    s.close()  # should not raise
    assert s._client is None
