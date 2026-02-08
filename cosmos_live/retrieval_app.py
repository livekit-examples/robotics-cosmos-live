"""Entry point for the Cosmos LiveKit voice agent.

Connects to a LiveKit room, starts the stream operator, and runs
a voice agent that can query feed content and control the stream.
"""

from __future__ import annotations

import logging
import sys

from livekit.agents import AgentSession, AutoSubscribe, RtcSession
from livekit.agents.voice import AgentServer
from livekit.plugins import silero

from cosmos_control.agent import CosmosAgent
from cosmos_control.operator import FFmpegStreamOperator
from cosmos_live.config import Config
from cosmos_storage import MilvusVectorStore

logger = logging.getLogger("cosmos_live.agent")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: RtcSession) -> None:
    _setup_logging()

    config = Config.from_yaml()
    logger.info("Loaded config for room=%s", config.livekit.room)

    vector_store = MilvusVectorStore(
        db_path=config.milvus.db_path,
        collection_name=config.milvus.collection_name,
        embedding_model=config.milvus.embedding_model,
    )

    operator: FFmpegStreamOperator | None = None
    if config.stream is not None:
        operator = FFmpegStreamOperator(config.stream, config.livekit)
        await operator.start()
        logger.info("Stream operator started")

    agent = CosmosAgent(
        vector_store=vector_store,
        operator=operator,
        instructions=config.agent.instructions,
    )

    session = AgentSession(
        stt=config.agent.stt,
        llm=config.agent.llm,
        tts=config.agent.tts,
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    await session.generate_reply(
        instructions="Greet the user briefly and let them know you can help monitor feeds and control the stream."
    )


if __name__ == "__main__":
    server.run()
