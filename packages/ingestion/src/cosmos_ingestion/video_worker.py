from __future__ import annotations

from cosmos_core import Frame, FrameBuffer, VisionModel, VectorStore


class VideoWorker:
    """Receives frames for a single video track, buffers, samples, and analyzes."""

    def __init__(
        self,
        track_id: str,
        buffer: FrameBuffer | None = None,
        vision: VisionModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.track_id = track_id
        self._buffer = buffer
        self._vision = vision
        self._vector_store = vector_store

    def push_frame(self, frame: Frame) -> None:
        """Push a single frame into the buffer."""
        if self._buffer is not None:
            self._buffer.push(frame)

    async def process(self, sample_count: int, prompt: str) -> None:
        """Sample frames from the buffer, analyze, and store results."""
        if self._buffer is None or self._vision is None:
            return

        frames = self._buffer.sample(sample_count)
        self._buffer.flush()

        if not frames:
            return

        result = await self._vision.analyze(frames, prompt)

        if self._vector_store is not None:
            from cosmos_core import Document

            doc = Document(
                content=result.answer,
                track_id=self.track_id,
                kind="analysis",
                timestamp=result.timestamp,
            )
            await self._vector_store.insert(doc)
