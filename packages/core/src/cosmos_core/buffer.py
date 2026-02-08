from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_core.types import Frame


class FrameBuffer(ABC):
    """Accumulates video frames and yields sampled subsets."""

    @abstractmethod
    def push(self, frame: Frame) -> None:
        """Append a single frame to the buffer."""

    @abstractmethod
    def sample(self, n: int) -> list[Frame]:
        """Return *n* evenly-spaced frames from the buffer."""

    @abstractmethod
    def flush(self) -> None:
        """Discard all buffered frames."""
