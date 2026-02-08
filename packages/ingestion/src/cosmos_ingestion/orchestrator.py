from __future__ import annotations

from typing import TYPE_CHECKING

from cosmos_core import VisionModel, VectorStore
from cosmos_ingestion.video_worker import VideoWorker
from cosmos_ingestion.audio_worker import AudioWorker

if TYPE_CHECKING:
    from cosmos_live.config import LiveKitConfig


class Orchestrator:
    """Connects to a LiveKit room and spawns workers per track."""

    def __init__(
        self,
        config: LiveKitConfig,
        vision: VisionModel,
        vector_store: VectorStore,
    ) -> None:
        self._config = config
        self._vision = vision
        self._vector_store = vector_store
        self._video_workers: dict[str, VideoWorker] = {}
        self._audio_workers: dict[str, AudioWorker] = {}

    async def run(self) -> None:
        """Connect to a LiveKit room and subscribe to tracks."""
        # TODO: use livekit SDK with self._config
        raise NotImplementedError

    async def _on_track(self, track_id: str, kind: str) -> None:
        if kind == "video":
            worker = VideoWorker(
                track_id,
                vision=self._vision,
                vector_store=self._vector_store,
            )
            self._video_workers[track_id] = worker
        elif kind == "audio":
            worker = AudioWorker(track_id)
            self._audio_workers[track_id] = worker
