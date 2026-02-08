from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, SecretStr


class LiveKitConfig(BaseModel):
    url: str
    api_key: str
    api_secret: SecretStr
    room: str


class VLLMConfig(BaseModel):
    base_url: str
    model: str
    api_key: SecretStr | None = None


class MilvusConfig(BaseModel):
    db_path: str = "cosmos.db"
    collection_name: str = "cosmos_documents"
    embedding_model: str = "all-MiniLM-L6-v2"


class OperatorConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    font_path: str = ""
    font_size: int = 36
    video_bitrate: str = "4500k"
    audio_bitrate: str = "128k"
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    gop_size: int = 60
    preset: str = "veryfast"
    placeholder_color: tuple[int, int, int] = (30, 30, 30)


class StreamConfig(BaseModel):
    rtmp_url: str
    stream_key: SecretStr
    operator: OperatorConfig = OperatorConfig()


class AgentConfig(BaseModel):
    stt: str = "deepgram/nova-3:multi"
    llm: str = "openai/gpt-4.1-mini"
    tts: str = "cartesia/sonic-3"
    instructions: str = (
        "You are Cosmos, an intelligent live stream director. "
        "You monitor multiple live video feeds and control a stream output.\n\n"
        "You can:\n"
        "- Search feed content to answer questions about what's happening or what happened\n"
        "- Switch the active feed being streamed\n"
        "- Set text overlays on the stream output\n"
        "- List available feeds\n\n"
        "When asked about feed content, use query for semantic/meaning-based lookups or search for field-based filtering (by feed, time, kind).\n"
        "When asked to switch feeds, first search for the right feed if not specified, then use switch_feed.\n"
        "Keep responses concise since you're a voice assistant."
    )


class VideoWorkerConfig(BaseModel):
    buffer_size: int = 60
    sample_count: int = 20
    target_fps: int = 10
    prompt: str = "Describe what is happening in these video frames."


class Config(BaseModel):
    livekit: LiveKitConfig
    vllm: VLLMConfig
    milvus: MilvusConfig
    stream: StreamConfig | None = None
    video_worker: VideoWorkerConfig = VideoWorkerConfig()
    agent: AgentConfig = AgentConfig()

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Config:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
