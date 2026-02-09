from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from cosmos_core import AnalysisResult, Document
from cosmos_core.ring_buffer import RingFrameBuffer
from cosmos_ingestion.video_worker import VideoWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(value: int = 0) -> np.ndarray:
    """Create a small dummy frame filled with *value*."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# RingFrameBuffer
# ---------------------------------------------------------------------------


class TestRingFrameBuffer:
    def test_push_and_length(self):
        buf = RingFrameBuffer(capacity=5)
        for i in range(3):
            buf.push(_frame(i))
        assert not buf.is_full

    def test_is_full(self):
        buf = RingFrameBuffer(capacity=3)
        for i in range(3):
            buf.push(_frame(i))
        assert buf.is_full

    def test_sample_evenly_spaced(self):
        buf = RingFrameBuffer(capacity=10)
        for i in range(10):
            buf.push(_frame(i))

        sampled = buf.sample(3)
        assert len(sampled) == 3
        # With linspace(0,9,3) -> indices [0, 4, 9]
        np.testing.assert_array_equal(sampled[0], _frame(0))
        np.testing.assert_array_equal(sampled[1], _frame(4))
        np.testing.assert_array_equal(sampled[2], _frame(9))

    def test_sample_when_empty(self):
        buf = RingFrameBuffer(capacity=5)
        assert buf.sample(3) == []

    def test_sample_count_exceeds_buffer(self):
        buf = RingFrameBuffer(capacity=10)
        for i in range(3):
            buf.push(_frame(i))
        sampled = buf.sample(10)
        assert len(sampled) == 3

    def test_flush(self):
        buf = RingFrameBuffer(capacity=5)
        for i in range(5):
            buf.push(_frame(i))
        assert buf.is_full
        buf.flush()
        assert not buf.is_full
        assert buf.sample(1) == []


# ---------------------------------------------------------------------------
# VideoWorker
# ---------------------------------------------------------------------------


class TestVideoWorker:
    @pytest.mark.asyncio
    async def test_push_frame_triggers_processing_when_full(self):
        vision = AsyncMock()
        vision.analyze.return_value = AnalysisResult(
            answer="test", timestamp=1.0
        )
        store = AsyncMock()

        worker = VideoWorker(
            track_id="t1",
            buffer_size=3,
            sample_count=2,
            vision=vision,
            vector_store=store,
        )

        worker.push_frame(_frame(0), timestamp=0.1)
        worker.push_frame(_frame(1), timestamp=0.2)
        # Buffer not full yet — no task
        assert worker._task is None

        worker.push_frame(_frame(2), timestamp=0.3)
        # Buffer is now full — task created
        assert worker._task is not None
        await worker._task

        vision.analyze.assert_awaited_once()
        frames_arg = vision.analyze.call_args[0][0]
        assert len(frames_arg) == 2  # sample_count=2

        store.insert.assert_awaited_once()
        doc: Document = store.insert.call_args[0][0]
        assert doc.content == "test"
        assert doc.track_id == "t1"
        assert doc.kind == "video"
        assert doc.timestamp == 0.3  # frame timestamp, not result timestamp

    @pytest.mark.asyncio
    async def test_buffer_flushed_after_trigger(self):
        vision = AsyncMock()
        vision.analyze.return_value = AnalysisResult(
            answer="a", timestamp=1.0
        )

        worker = VideoWorker(
            track_id="t1",
            buffer_size=2,
            sample_count=1,
            vision=vision,
            vector_store=AsyncMock(),
        )

        worker.push_frame(_frame(0))
        worker.push_frame(_frame(1))
        await worker._task

        # Buffer should be flushed — can accept new frames without immediate trigger
        assert not worker._buffer.is_full

    @pytest.mark.asyncio
    async def test_no_vision_skips_processing(self):
        worker = VideoWorker(
            track_id="t1",
            buffer_size=2,
            sample_count=1,
            vision=None,
            vector_store=AsyncMock(),
        )

        worker.push_frame(_frame(0))
        worker.push_frame(_frame(1))
        await worker._task  # should complete without error

    @pytest.mark.asyncio
    async def test_no_vector_store_skips_insert(self):
        vision = AsyncMock()
        vision.analyze.return_value = AnalysisResult(
            answer="ok", timestamp=1.0
        )

        worker = VideoWorker(
            track_id="t1",
            buffer_size=2,
            sample_count=1,
            vision=vision,
            vector_store=None,
        )

        worker.push_frame(_frame(0))
        worker.push_frame(_frame(1))
        await worker._task

        vision.analyze.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_waits_for_inflight_task(self):
        event = asyncio.Event()

        async def slow_analyze(frames, prompt):
            await event.wait()
            return AnalysisResult(answer="done", timestamp=1.0)

        vision = AsyncMock()
        vision.analyze.side_effect = slow_analyze

        worker = VideoWorker(
            track_id="t1",
            buffer_size=2,
            sample_count=1,
            vision=vision,
            vector_store=AsyncMock(),
        )

        worker.push_frame(_frame(0))
        worker.push_frame(_frame(1))

        assert not worker._task.done()

        # Unblock the analyze call
        event.set()
        await worker.stop()

        assert worker._task.done()

    @pytest.mark.asyncio
    async def test_stop_noop_when_no_task(self):
        worker = VideoWorker(track_id="t1", buffer_size=5)
        await worker.stop()  # should not raise
