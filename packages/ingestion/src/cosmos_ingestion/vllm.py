from __future__ import annotations

import base64
import logging
import time

import cv2
import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from cosmos_core import Frame, AnalysisResult, VisionModel
from cosmos_live.config import VLLMConfig


MAX_FRAME_DIM = 512


def _encode_frame(frame: Frame, quality: int = 75) -> str:
    """Resize (if needed), JPEG-encode, and return a base64 data URI."""
    h, w = frame.shape[:2]
    if max(h, w) > MAX_FRAME_DIM:
        scale = MAX_FRAME_DIM / max(h, w)
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to encode frame as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class VLLMVisionModel(VisionModel):
    """vLLM-backed vision model implementation."""

    def __init__(self, config: VLLMConfig) -> None:
        self._config = config
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        """Lazy-create the AsyncOpenAI client."""
        if self._client is None:
            api_key = (
                self._config.api_key.get_secret_value()
                if self._config.api_key
                else "EMPTY"
            )
            self._client = AsyncOpenAI(
                base_url=self._config.base_url + "/v1",
                api_key=api_key,
            )
        return self._client

    async def analyze(self, frames: list[Frame], prompt: str) -> AnalysisResult:
        client = self._ensure_client()

        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": _encode_frame(frame)}}
            for frame in frames
        ]
        content.append({"type": "text", "text": prompt})

        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": content}],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        metadata: dict = {}
        if response.model:
            metadata["model"] = response.model
        if response.usage:
            metadata["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info(
            "vLLM analyze frames=%d elapsed=%.0fms tokens=%s",
            len(frames),
            elapsed_ms,
            metadata.get("usage", {}).get("total_tokens", "n/a"),
        )

        return AnalysisResult(
            answer=choice.message.content or "",
            metadata=metadata,
            timestamp=time.time(),
        )

    async def close(self) -> None:
        """Release client resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
