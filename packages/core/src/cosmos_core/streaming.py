from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_core.types import Overlay


class StreamOperator(ABC):
    """Controls the live output stream (e.g. YouTube, RTMP)."""

    @abstractmethod
    async def set_feed(self, feed_id: str) -> None:
        """Switch the active video feed to *feed_id*."""

    @abstractmethod
    async def set_overlay(self, overlay: Overlay) -> None:
        """Apply an overlay to the live output."""

    @abstractmethod
    async def start(self) -> None:
        """Begin streaming."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop streaming."""
