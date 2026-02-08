from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_core.types import AnalysisResult, Frame


class VisionModel(ABC):
    """Processes a batch of video frames with a text prompt."""

    @abstractmethod
    async def analyze(self, frames: list[Frame], prompt: str) -> AnalysisResult:
        """Analyze *frames* according to *prompt* and return a structured result."""
