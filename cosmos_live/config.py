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
    uri: str
    token: SecretStr | None = None


class StreamConfig(BaseModel):
    rtmp_url: str
    stream_key: SecretStr


class Config(BaseModel):
    livekit: LiveKitConfig
    vllm: VLLMConfig
    milvus: MilvusConfig
    stream: StreamConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Config:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
