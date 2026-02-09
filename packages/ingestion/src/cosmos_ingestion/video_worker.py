from __future__ import annotations

import asyncio

from cosmos_core import Document, Frame, RingFrameBuffer, VisionModel, VectorStore


class VideoWorker:
    """Receives frames for a single video track, buffers, samples, and analyzes."""

    def __init__(
        self,
        track_id: str,
        buffer_size: int = 60,
        sample_count: int = 20,
        prompt: str = "Describe what is happening in these video frames.",
        vision: VisionModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.track_id = track_id
        self._buffer = RingFrameBuffer(capacity=buffer_size)
        self._sample_count = sample_count
        self._prompt = prompt
        self._vision = vision
        self._vector_store = vector_store
        self._task: asyncio.Task[None] | None = None

    def push_frame(self, frame: Frame, timestamp: float = 0.0) -> None:
        """Push a single frame into the buffer; auto-processes when full."""
        self._buffer.push(frame)
        if self._buffer.is_full:
            frames = self._buffer.sample(self._sample_count)
            self._buffer.flush()
            self._task = asyncio.get_running_loop().create_task(
                self._process_chunk(frames, timestamp)
            )

    async def _process_chunk(self, frames: list[Frame], timestamp: float) -> None:
        if self._vision is None:
            return

        result = await self._vision.analyze(frames, self._prompt)

        if self._vector_store is not None:
            doc = Document(
                content=result.answer,
                track_id=self.track_id,
                kind="video",
                timestamp=timestamp,
            )
            await self._vector_store.insert(doc)

    async def stop(self) -> None:
        """Wait for any in-flight processing task to complete."""
        if self._task is not None and not self._task.done():
            await self._task
