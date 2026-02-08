from __future__ import annotations

import numpy as np

from cosmos_core.buffer import FrameBuffer
from cosmos_core.types import Frame


class RingFrameBuffer(FrameBuffer):
    """Fixed-capacity list buffer that accumulates frames and yields sampled subsets."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._frames: list[Frame] = []

    @property
    def is_full(self) -> bool:
        return len(self._frames) >= self._capacity

    def push(self, frame: Frame) -> None:
        self._frames.append(frame)

    def sample(self, n: int) -> list[Frame]:
        if not self._frames:
            return []
        n = min(n, len(self._frames))
        indices = np.linspace(0, len(self._frames) - 1, n, dtype=int)
        return [self._frames[i] for i in indices]

    def flush(self) -> None:
        self._frames.clear()
