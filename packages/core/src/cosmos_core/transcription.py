from __future__ import annotations

from abc import ABC, abstractmethod

from cosmos_core.types import AudioSegment


class Transcriber(ABC):
    """Converts audio segments to text."""

    @abstractmethod
    async def transcribe(self, audio: AudioSegment) -> str:
        """Return a text transcription of the given audio."""
