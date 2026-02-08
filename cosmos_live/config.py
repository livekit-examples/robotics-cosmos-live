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


class StreamConfig(BaseModel):
    rtmp_url: str
    stream_key: SecretStr


class VideoWorkerConfig(BaseModel):
    buffer_size: int = 60
    sample_count: int = 20
    prompt: str = "Describe what is happening in these video frames."


class Config(BaseModel):
    livekit: LiveKitConfig
    vllm: VLLMConfig
    milvus: MilvusConfig
    stream: StreamConfig | None = None
    video_worker: VideoWorkerConfig = VideoWorkerConfig()

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Config:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
