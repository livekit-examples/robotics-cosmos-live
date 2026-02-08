from __future__ import annotations


class Publisher:
    """Captures video/audio from a source and publishes to a LiveKit room."""

    def __init__(self, room: str, source: str) -> None:
        self._room = room
        self._source = source  # e.g. device path, RTSP URL, screen capture

    async def run(self) -> None:
        """Connect to the room and start publishing tracks."""
        # TODO: use livekit SDK to join room and publish video/audio tracks
        raise NotImplementedError
