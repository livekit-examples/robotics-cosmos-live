"""Entry point that wires ingestion + control together."""

from __future__ import annotations

import asyncio

from cosmos_live.config import Config
from cosmos_storage import InMemoryVectorStore
from cosmos_ingestion.vllm import VLLMVisionModel
from cosmos_ingestion.orchestrator import Orchestrator
from cosmos_control.agent import ControlAgent
from cosmos_control.operator import FFmpegStreamOperator


async def main() -> None:
    config = Config.from_yaml()

    # Shared store — ingestion writes, control reads
    vector_store = InMemoryVectorStore()

    # Ingestion pipeline
    vision = VLLMVisionModel()
    orchestrator = Orchestrator(
        config=config.livekit,
        vision=vision,
        vector_store=vector_store,
    )

    # Control pipeline
    stream_operator = FFmpegStreamOperator()
    agent = ControlAgent(
        vector_store=vector_store,
        stream_operator=stream_operator,
    )

    # Run both flows concurrently
    await asyncio.gather(
        orchestrator.run(),
        agent.run(),
    )


if __name__ == "__main__":
    asyncio.run(main())
