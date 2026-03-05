from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from cosmos_utils import generate_livekit_token
from livekit import rtc

if TYPE_CHECKING:
    from cosmos_live.config import LiveKitConfig

logger = logging.getLogger(__name__)


class Publisher:
    """Captures video/audio from a source and publishes to a LiveKit room."""

    def __init__(
        self,
        config: LiveKitConfig,
        source: str,
        *,
        room_name: str | None = None,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        identity: str = "publisher",
    ) -> None:
        self._config = config
        self._room_name = room_name or config.room
        self._source = source
        self._fps = fps
        self._width = width
        self._height = height
        self._identity = identity
        self._running = False

    async def run(self) -> None:
        """Connect to the room, publish a video track, and start capturing."""
        video_source = rtc.VideoSource(self._width, self._height)
        track = rtc.LocalVideoTrack.create_video_track("camera", video_source)

        room = rtc.Room()
        token = generate_livekit_token(
            self._config, self._identity, self._room_name
        )

        logger.info(
            "Connecting to %s room=%s as %s",
            self._config.url,
            self._room_name,
            self._identity,
        )
        await room.connect(self._config.url, token)
        logger.info("Connected, publishing video track")

        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(
                max_framerate=self._fps,
                max_bitrate=5_000_000,
            ),
        )
        await room.local_participant.publish_track(track, options)

        try:
            await self._capture_loop(video_source)
        finally:
            await room.disconnect()
            logger.info("Disconnected from room")

    async def _capture_loop(self, video_source: rtc.VideoSource) -> None:
        """Read frames from the video source and push them to LiveKit."""
        # Support integer device index (e.g. "0" for webcam) or URL string
        try:
            device = int(self._source)
        except ValueError:
            device = self._source

        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self._source}")

        frame_interval = 1.0 / self._fps
        self._running = True
        logger.info(
            "Capture loop started: source=%s fps=%d %dx%d",
            self._source,
            self._fps,
            self._width,
            self._height,
        )

        frame_count = 0
        try:
            while self._running:
                ok, bgr = await asyncio.to_thread(cap.read)
                if not ok:
                    logger.warning("Failed to read frame, stopping capture")
                    break

                bgr = cv2.resize(bgr, (self._width, self._height))
                rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)

                frame = rtc.VideoFrame(
                    self._width,
                    self._height,
                    rtc.VideoBufferType.RGBA,
                    rgba.tobytes(),
                )
                video_source.capture_frame(frame)
                frame_count += 1
                if frame_count % (self._fps * 5) == 0:
                    logger.info("Published %d frames", frame_count)

                await asyncio.sleep(frame_interval)
        finally:
            cap.release()
            logger.info("Capture loop stopped")

    def stop(self) -> None:
        """Signal the capture loop to exit cleanly."""
        self._running = False
