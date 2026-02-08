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
    collection_name: str = "cosmos_documents"
    embedding_model: str = "all-MiniLM-L6-v2"


class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password: SecretStr


class StreamConfig(BaseModel):
    rtmp_url: str
    stream_key: SecretStr


class Config(BaseModel):
    livekit: LiveKitConfig
    vllm: VLLMConfig
    milvus: MilvusConfig
    neo4j: Neo4jConfig
    stream: StreamConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Config:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
