from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_core.types import Overlay


class StreamOperator(ABC):
    """Controls the live output stream (e.g. YouTube, RTMP)."""

    @abstractmethod
    async def set_feed(self, feed_id: str) -> None:
        """Switch the active video/audio feed by participant identity or track ID."""

    @abstractmethod
    async def set_overlay(self, slot: str, overlay: Overlay) -> None:
        """Set a named overlay slot (e.g. 'lower_third', 'title', 'banner')."""

    @abstractmethod
    async def clear_overlay(self, slot: str) -> None:
        """Remove the overlay from a named slot."""

    @abstractmethod
    async def start(self) -> None:
        """Begin streaming."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop streaming."""
