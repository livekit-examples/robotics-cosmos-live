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
    uri: str = "cosmos.db"
    collection_name: str = "cosmos_documents"
    embedding_model: str = "all-MiniLM-L6-v2"


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection_name: str = "cosmos_documents"
    embedding_model: str = "all-MiniLM-L6-v2"


class OperatorConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    placeholder_color: tuple[int, int, int] = (30, 30, 30)
    display_port: int = 9090


class AgentConfig(BaseModel):
    stt: str = "deepgram/nova-3:multi"
    llm: str = "openai/gpt-4.1-mini"
    tts: str = "cartesia/sonic-3"
    instructions: str = (
        "You are Cosmos, an intelligent live stream director. "
        "You monitor multiple live video feeds and control a stream output.\n\n"
        "## Tool selection rules\n\n"
        "You have two search tools. Picking the right one is critical:\n\n"
        "### Use `search` (filter-based, most-recent-first) for:\n"
        "- Current state questions: \"what's happening?\", \"what's going on?\", \"describe the feed\"\n"
        "- Recent activity: \"what just happened?\", \"any updates?\"\n"
        "- Time-bounded queries: \"what happened in the last 5 minutes?\"\n"
        "- Specific feed status: \"what's cam-1 showing?\", \"describe feed X\"\n"
        "- Transcript/speech lookups: \"what did someone say?\"\n"
        "- ANY question about the present or recent past\n\n"
        "### Use `query` (semantic search) ONLY for:\n"
        "- Finding feeds/content by description: \"which feed has a dog?\", \"find the one with cooking\"\n"
        "- Searching for a specific event by meaning: \"was there ever a fire?\"\n"
        "- When the user is looking for something by what it looks like or means, not by when it happened\n\n"
        "### Key principle: if the question is about NOW or RECENT, always use `search`. "
        "Only use `query` when the user is searching for something by its content/meaning.\n\n"
        "### Time-based searches\n"
        "For questions about a specific time window (e.g. \"last 5 minutes\"), "
        "call `get_current_time` first, subtract the duration, and pass the result as `since`.\n\n"
        "## Other tools\n"
        "- `switch_feed`: Switch the active stream. If the user names a feed vaguely, "
        "use `list_feeds` or `search` to find the right feed_id first.\n"
        "- `set_overlay` / `clear_overlay`: Manage text overlays on the stream.\n"
        "- `list_feeds`: List all available feed identities.\n\n"
        "## Style\n"
        "Keep responses concise and short — you are a voice assistant. "
        "Summarize search results naturally; don't read raw data back to the user."
    )


class VideoWorkerConfig(BaseModel):
    buffer_size: int = 60
    sample_count: int = 20
    target_fps: int = 10
    prompt: str = "Describe what is happening in these video frames."


class Config(BaseModel):
    livekit: LiveKitConfig
    vllm: VLLMConfig
    qdrant: QdrantConfig
    operator: OperatorConfig | None = None
    video_worker: VideoWorkerConfig = VideoWorkerConfig()
    agent: AgentConfig = AgentConfig()

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Config:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
