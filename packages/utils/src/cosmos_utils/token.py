from __future__ import annotations

from typing import TYPE_CHECKING

from livekit import api

if TYPE_CHECKING:
    from cosmos_live.config import LiveKitConfig


def generate_livekit_token(
    config: LiveKitConfig,
    identity: str,
    room: str,
) -> str:
    """Create a JWT access token with publish grants for a LiveKit room."""
    token = api.AccessToken(
        config.api_key,
        config.api_secret.get_secret_value(),
    )
    token.with_identity(identity)
    token.with_grants(
        api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
        )
    )
    return token.to_jwt()
