from __future__ import annotations

from cosmos_core import Frame, AnalysisResult, VisionModel


class VLLMVisionModel(VisionModel):
    """vLLM-backed vision model implementation."""

    async def analyze(self, frames: list[Frame], prompt: str) -> AnalysisResult:
        raise NotImplementedError
