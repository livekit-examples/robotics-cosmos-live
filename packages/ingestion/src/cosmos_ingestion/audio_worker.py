from __future__ import annotations

from cosmos_core import AudioSegment, Transcriber, VectorStore, Document


class AudioWorker:
    """Receives audio for a single track, transcribes, and stores results."""

    def __init__(
        self,
        track_id: str,
        transcriber: Transcriber | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.track_id = track_id
        self._transcriber = transcriber
        self._vector_store = vector_store

    async def push_audio(self, audio: AudioSegment) -> None:
        """Transcribe an audio segment and store the result."""
        if self._transcriber is None:
            return

        text = await self._transcriber.transcribe(audio)

        doc = Document(
            content=text,
            track_id=self.track_id,
            kind="audio",
        )

        if self._vector_store is not None:
            await self._vector_store.insert(doc)
