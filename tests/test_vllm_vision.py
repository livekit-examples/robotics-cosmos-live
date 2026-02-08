from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from openai import APIError

from cosmos_live.config import Config, VLLMConfig
from cosmos_ingestion.vllm import VLLMVisionModel, _encode_frame

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_vllm_config_defaults():
    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    assert cfg.base_url == "http://localhost:8000"
    assert cfg.model == "llava"
    assert cfg.api_key is None


def test_vllm_config_with_api_key():
    cfg = VLLMConfig(
        base_url="http://localhost:8000",
        model="llava",
        api_key="secret-key",
    )
    assert cfg.api_key.get_secret_value() == "secret-key"


# ---------------------------------------------------------------------------
# Constructor — no eager client
# ---------------------------------------------------------------------------


def test_constructor_no_eager_client():
    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)
    assert model._client is None


# ---------------------------------------------------------------------------
# _encode_frame
# ---------------------------------------------------------------------------


def test_encode_frame_success():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _encode_frame(frame)
    assert result.startswith("data:image/jpeg;base64,")
    # Verify the base64 portion is valid
    import base64
    b64_data = result.split(",", 1)[1]
    decoded = base64.b64decode(b64_data)
    assert len(decoded) > 0


@patch("cosmos_ingestion.vllm.cv2.imencode", return_value=(False, None))
def test_encode_frame_failure(mock_imencode):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Failed to encode frame as JPEG"):
        _encode_frame(frame)


# ---------------------------------------------------------------------------
# _ensure_client
# ---------------------------------------------------------------------------


@patch("cosmos_ingestion.vllm.AsyncOpenAI", autospec=True)
def test_ensure_client_without_api_key(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)
    client = model._ensure_client()

    mock_openai_cls.assert_called_once_with(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    assert client is mock_client


@patch("cosmos_ingestion.vllm.AsyncOpenAI", autospec=True)
def test_ensure_client_with_api_key(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    cfg = VLLMConfig(
        base_url="http://localhost:8000", model="llava", api_key="my-key"
    )
    model = VLLMVisionModel(config=cfg)
    client = model._ensure_client()

    mock_openai_cls.assert_called_once_with(
        base_url="http://localhost:8000/v1",
        api_key="my-key",
    )
    assert client is mock_client


@patch("cosmos_ingestion.vllm.AsyncOpenAI", autospec=True)
def test_ensure_client_singleton(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)
    c1 = model._ensure_client()
    c2 = model._ensure_client()

    assert c1 is c2
    mock_openai_cls.assert_called_once()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _make_mock_response(content="test answer", model="llava", usage=True):
    """Build a mock ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    if usage:
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
    else:
        response.usage = None
    return response


@pytest.mark.asyncio
@patch("cosmos_ingestion.vllm._encode_frame", return_value="data:image/jpeg;base64,AAAA")
@patch("cosmos_ingestion.vllm.AsyncOpenAI", autospec=True)
async def test_analyze_single_frame(mock_openai_cls, mock_encode):
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response()
    )
    mock_openai_cls.return_value = mock_client

    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = await model.analyze([frame], "describe this image")

    assert result.answer == "test answer"
    assert result.metadata["model"] == "llava"
    assert result.metadata["usage"]["total_tokens"] == 15
    assert result.timestamp > 0

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llava"
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    content_parts = messages[0]["content"]
    assert len(content_parts) == 2  # 1 image + 1 text
    assert content_parts[0]["type"] == "image_url"
    assert content_parts[1]["type"] == "text"
    assert content_parts[1]["text"] == "describe this image"


@pytest.mark.asyncio
@patch("cosmos_ingestion.vllm._encode_frame", return_value="data:image/jpeg;base64,AAAA")
@patch("cosmos_ingestion.vllm.AsyncOpenAI", autospec=True)
async def test_analyze_multiple_frames(mock_openai_cls, mock_encode):
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response()
    )
    mock_openai_cls.return_value = mock_client

    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    result = await model.analyze(frames, "compare these frames")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    content_parts = call_kwargs["messages"][0]["content"]
    assert len(content_parts) == 4  # 3 images + 1 text
    assert all(p["type"] == "image_url" for p in content_parts[:3])
    assert content_parts[3]["type"] == "text"


@pytest.mark.asyncio
@patch("cosmos_ingestion.vllm._encode_frame", return_value="data:image/jpeg;base64,AAAA")
@patch("cosmos_ingestion.vllm.AsyncOpenAI", autospec=True)
async def test_analyze_missing_usage(mock_openai_cls, mock_encode):
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response(usage=False)
    )
    mock_openai_cls.return_value = mock_client

    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = await model.analyze([frame], "describe this")

    assert result.answer == "test answer"
    assert "usage" not in result.metadata
    assert result.metadata["model"] == "llava"


# ---------------------------------------------------------------------------
# API error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("cosmos_ingestion.vllm._encode_frame", return_value="data:image/jpeg;base64,AAAA")
@patch("cosmos_ingestion.vllm.AsyncOpenAI", autospec=True)
async def test_analyze_api_error_propagates(mock_openai_cls, mock_encode):
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
    )
    mock_openai_cls.return_value = mock_client

    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(APIError):
        await model.analyze([frame], "describe this")


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_releases_client():
    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)

    mock_client = AsyncMock()
    model._client = mock_client

    await model.close()

    mock_client.close.assert_awaited_once()
    assert model._client is None


@pytest.mark.asyncio
async def test_close_safe_when_no_client():
    cfg = VLLMConfig(base_url="http://localhost:8000", model="llava")
    model = VLLMVisionModel(config=cfg)

    await model.close()  # should not raise
    assert model._client is None


# ---------------------------------------------------------------------------
# Integration tests — require a running vLLM server + config.yaml
# ---------------------------------------------------------------------------


@pytest.fixture
def vllm_config():
    """Load VLLMConfig from the project config.yaml."""
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    config = Config.from_yaml(CONFIG_PATH)
    return config.vllm


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_analyze_single_frame(vllm_config):
    """Send a real frame to the vLLM server and verify a non-empty response."""
    model = VLLMVisionModel(config=vllm_config)
    try:
        # Create a simple test frame (red square on black background)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[50:174, 50:174] = [0, 0, 255]  # red square (BGR)

        result = await model.analyze([frame], "What do you see in this image?")

        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.timestamp > 0
        assert "model" in result.metadata
    finally:
        await model.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_analyze_multiple_frames(vllm_config):
    """Send multiple frames and verify the model responds."""
    model = VLLMVisionModel(config=vllm_config)
    try:
        frames = []
        for color in ([255, 0, 0], [0, 255, 0], [0, 0, 255]):
            frame = np.full((224, 224, 3), color, dtype=np.uint8)
            frames.append(frame)

        result = await model.analyze(frames, "Describe the differences between these images.")

        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
    finally:
        await model.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_client_reuse(vllm_config):
    """Verify the client is reused across multiple analyze calls."""
    model = VLLMVisionModel(config=vllm_config)
    try:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

        await model.analyze([frame], "What is this?")
        client_after_first = model._client

        await model.analyze([frame], "Describe this image.")
        client_after_second = model._client

        assert client_after_first is client_after_second
    finally:
        await model.close()
