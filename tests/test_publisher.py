from __future__ import annotations

import json
from base64 import b64decode

from cosmos_ingestion.publisher import Publisher
from cosmos_live.config import LiveKitConfig
from cosmos_utils import generate_livekit_token


def _make_config() -> LiveKitConfig:
    return LiveKitConfig(
        url="http://localhost:7880",
        api_key="devkey",
        api_secret="secret",
        room="test-room",
    )


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verification (test only)."""
    payload_b64 = token.split(".")[1]
    # Add padding
    payload_b64 += "=" * (-len(payload_b64) % 4)
    return json.loads(b64decode(payload_b64))


def test_generate_livekit_token():
    config = _make_config()
    token = generate_livekit_token(config, identity="test-user", room="my-room")

    payload = _decode_jwt_payload(token)
    assert payload["sub"] == "test-user"
    video = payload["video"]
    assert video["room"] == "my-room"
    assert video["roomJoin"] is True
    assert video["canPublish"] is True


def test_publisher_init():
    config = _make_config()
    pub = Publisher(config, source="0", identity="cam-1")

    assert pub._config is config
    assert pub._room_name == "test-room"
    assert pub._source == "0"
    assert pub._identity == "cam-1"
    assert pub._running is False


def test_publisher_init_room_override():
    config = _make_config()
    pub = Publisher(config, source="0", room_name="override-room")

    assert pub._room_name == "override-room"


def test_publisher_stop():
    config = _make_config()
    pub = Publisher(config, source="0")
    pub._running = True

    pub.stop()

    assert pub._running is False
