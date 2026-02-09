from __future__ import annotations

import pytest

from cosmos_core.types import Document
from cosmos_live.config import QdrantConfig
from cosmos_storage.qdrant import QdrantVectorStore


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_qdrant_config_defaults():
    cfg = QdrantConfig()
    assert cfg.url == "http://localhost:6333"
    assert cfg.collection_name == "cosmos_documents"
    assert cfg.embedding_model == "all-MiniLM-L6-v2"


def test_qdrant_config_custom():
    cfg = QdrantConfig(
        url="http://qdrant:6333",
        collection_name="my_docs",
        embedding_model="custom-model",
    )
    assert cfg.url == "http://qdrant:6333"
    assert cfg.collection_name == "my_docs"
    assert cfg.embedding_model == "custom-model"


# ---------------------------------------------------------------------------
# Fixture — fresh store per test with a unique collection name
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(request):
    """Create a QdrantVectorStore backed by a unique collection.

    Requires a running Qdrant server at localhost:6333.
    """
    collection = f"test_{request.node.name}"
    s = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name=collection,
    )
    yield s
    # Clean up: drop the test collection
    try:
        client = s._ensure_client()
        client.delete_collection(collection)
    except Exception:
        pass
    s.close()


# ---------------------------------------------------------------------------
# Constructor — no eager connections
# ---------------------------------------------------------------------------


def test_constructor_no_eager_connections():
    store = QdrantVectorStore()
    assert store._model is None
    assert store._client is None


# ---------------------------------------------------------------------------
# _embed — real embeddings
# ---------------------------------------------------------------------------


def test_embed_returns_384d_vectors(store: QdrantVectorStore):
    vecs = store._embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384


def test_embed_batch(store: QdrantVectorStore):
    vecs = store._embed(["foo", "bar", "baz"])
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 384


# ---------------------------------------------------------------------------
# insert + query — real round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_and_query_single_document(store: QdrantVectorStore):
    doc = Document(
        content="the quick brown fox jumps over the lazy dog",
        track_id="cam-1",
        kind="video",
        timestamp=1.0,
    )
    await store.insert(doc)

    results = await store.query("quick brown fox", top_k=1)
    assert len(results) == 1
    assert results[0].content == doc.content
    assert results[0].track_id == "cam-1"
    assert results[0].kind == "video"
    assert results[0].timestamp == 1.0


@pytest.mark.asyncio
async def test_insert_multiple_and_query_returns_most_relevant(store: QdrantVectorStore):
    docs = [
        Document(content="python programming language", track_id="t1", kind="audio", timestamp=1.0),
        Document(content="the weather is sunny today", track_id="t2", kind="video", timestamp=2.0),
        Document(content="machine learning and artificial intelligence", track_id="t3", kind="audio", timestamp=3.0),
    ]
    for doc in docs:
        await store.insert(doc)

    results = await store.query("deep learning AI", top_k=2)
    assert len(results) == 2
    # The ML/AI doc should be the top result
    assert results[0].content == "machine learning and artificial intelligence"


@pytest.mark.asyncio
async def test_query_empty_collection_returns_empty(store: QdrantVectorStore):
    results = await store.query("anything", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_query_top_k_limits_results(store: QdrantVectorStore):
    for i in range(5):
        await store.insert(Document(content=f"document number {i}", track_id="t", kind="k", timestamp=float(i)))

    results = await store.query("document", top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# search — filter-based with correct ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_most_recent_first(store: QdrantVectorStore):
    """Insert 20 docs and verify search(top_k=1) returns the genuinely most recent one."""
    for i in range(20):
        await store.insert(
            Document(
                content=f"event at time {i}",
                track_id="cam-1",
                kind="video",
                timestamp=float(i),
            )
        )

    results = await store.search(top_k=1)
    assert len(results) == 1
    assert results[0].timestamp == 19.0
    assert results[0].content == "event at time 19"


@pytest.mark.asyncio
async def test_search_filters_by_kind(store: QdrantVectorStore):
    await store.insert(Document(content="a audio", track_id="t1", kind="audio", timestamp=1.0))
    await store.insert(Document(content="an video", track_id="t1", kind="video", timestamp=2.0))

    results = await store.search(kind="audio")
    assert len(results) == 1
    assert results[0].kind == "audio"


@pytest.mark.asyncio
async def test_search_filters_by_track_id(store: QdrantVectorStore):
    await store.insert(Document(content="from cam-1", track_id="cam-1", kind="video", timestamp=1.0))
    await store.insert(Document(content="from cam-2", track_id="cam-2", kind="video", timestamp=2.0))

    results = await store.search(track_id="cam-1")
    assert len(results) == 1
    assert results[0].track_id == "cam-1"


@pytest.mark.asyncio
async def test_search_filters_by_since(store: QdrantVectorStore):
    await store.insert(Document(content="old", track_id="t", kind="k", timestamp=1.0))
    await store.insert(Document(content="new", track_id="t", kind="k", timestamp=10.0))

    results = await store.search(since=5.0)
    assert len(results) == 1
    assert results[0].content == "new"


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_releases_client(store: QdrantVectorStore):
    store._ensure_client()
    assert store._client is not None

    store.close()
    assert store._client is None


def test_close_safe_when_no_client():
    s = QdrantVectorStore()
    s.close()  # should not raise
    assert s._client is None
