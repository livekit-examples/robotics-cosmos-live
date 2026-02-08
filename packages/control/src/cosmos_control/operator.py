from __future__ import annotations

from cosmos_core import Overlay, StreamOperator


class FFmpegStreamOperator(StreamOperator):
    """Stream operator backed by FFmpeg for RTMP/YouTube output."""

    async def set_feed(self, feed_id: str) -> None:
        raise NotImplementedError

    async def set_overlay(self, overlay: Overlay) -> None:
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError
