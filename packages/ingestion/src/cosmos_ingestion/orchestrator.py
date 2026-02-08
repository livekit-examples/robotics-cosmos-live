from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from livekit import rtc

from cosmos_core import VisionModel, VectorStore
from cosmos_ingestion.video_worker import VideoWorker

# from cosmos_ingestion.audio_worker import AudioWorker
from cosmos_utils import generate_livekit_token

if TYPE_CHECKING:
    from cosmos_live.config import LiveKitConfig, VideoWorkerConfig

logger = logging.getLogger(__name__)

DEFAULT_TARGET_FPS = 10


class Orchestrator:
    """Connects to a LiveKit room and spawns workers per track."""

    def __init__(
        self,
        config: LiveKitConfig,
        vision: VisionModel,
        vector_store: VectorStore,
        video_worker_config: VideoWorkerConfig | None = None,
    ) -> None:
        self._config = config
        self._vision = vision
        self._vector_store = vector_store
        self._video_worker_config = video_worker_config
        target_fps = (
            video_worker_config.target_fps
            if video_worker_config is not None
            else DEFAULT_TARGET_FPS
        )
        self._frame_interval = 1.0 / target_fps
        self._video_workers: dict[str, VideoWorker] = {}
        # self._audio_workers: dict[str, AudioWorker] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def run(self) -> None:
        """Connect to a LiveKit room and subscribe to tracks."""
        room = rtc.Room()
        token = generate_livekit_token(
            self._config, "orchestrator", self._config.room
        )

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            identity = participant.identity
            logger.info(
                "Track subscribed: %s kind=%s from %s",
                track.sid,
                track.kind,
                identity,
            )
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._on_video_track(identity, track)
            # elif track.kind == rtc.TrackKind.KIND_AUDIO:
            #     self._on_audio_track(track_id, track)

        @room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            identity = participant.identity
            logger.info(
                "Track unsubscribed: %s from %s",
                track.sid,
                identity,
            )
            task = self._tasks.pop(identity, None)
            if task is not None:
                task.cancel()
            self._video_workers.pop(identity, None)

        logger.info("Connecting to %s room=%s", self._config.url, self._config.room)
        await room.connect(self._config.url, token)
        logger.info("Connected, waiting for tracks...")

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down orchestrator...")
            for task in self._tasks.values():
                task.cancel()
            for worker in self._video_workers.values():
                await worker.stop()
            await room.disconnect()
            logger.info("Disconnected")

    def _on_video_track(self, identity: str, track: rtc.Track) -> None:
        kwargs: dict = dict(
            vision=self._vision,
            vector_store=self._vector_store,
        )
        if self._video_worker_config is not None:
            kwargs["buffer_size"] = self._video_worker_config.buffer_size
            kwargs["sample_count"] = self._video_worker_config.sample_count
            kwargs["prompt"] = self._video_worker_config.prompt
        worker = VideoWorker(identity, **kwargs)
        self._video_workers[identity] = worker
        self._tasks[identity] = asyncio.create_task(
            self._consume_video(track, worker)
        )

    async def _consume_video(
        self, track: rtc.Track, worker: VideoWorker
    ) -> None:
        """Read frames from a LiveKit video track and push to the worker."""
        video_stream = rtc.VideoStream(track)
        last_frame_time = 0.0

        async for frame_event in video_stream:
            now = time.monotonic()
            if now - last_frame_time < self._frame_interval:
                continue
            last_frame_time = now

            frame = frame_event.frame
            rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
            arr = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                (rgb_frame.height, rgb_frame.width, 3)
            )
            
            worker.push_frame(arr, time.time())
