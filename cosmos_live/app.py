"""Entry point for the cosmos-live ingestion pipeline.

Connects to a LiveKit room, subscribes to video tracks, runs vision
analysis on buffered frames, and stores results in Milvus.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from cosmos_live.config import Config
from cosmos_storage import MilvusVectorStore
from cosmos_ingestion.vllm import VLLMVisionModel
from cosmos_ingestion.orchestrator import Orchestrator

logger = logging.getLogger("cosmos_live")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


async def main() -> None:
    _setup_logging()

    config = Config.from_yaml()
    logger.info("Loaded config for room=%s", config.livekit.room)

    vector_store = MilvusVectorStore(
        db_path=config.milvus.db_path,
        collection_name=config.milvus.collection_name,
        embedding_model=config.milvus.embedding_model,
    )

    vision = VLLMVisionModel(config=config.vllm)

    orchestrator = Orchestrator(
        config=config.livekit,
        vision=vision,
        vector_store=vector_store,
        video_worker_config=config.video_worker,
    )

    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    task = asyncio.create_task(orchestrator.run())

    logger.info("Ingestion pipeline started — press Ctrl+C to stop")
    await stop

    logger.info("Shutting down…")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    await vision.close()
    vector_store.close()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
