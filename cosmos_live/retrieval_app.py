"""Entry point for the Cosmos LiveKit voice agent.

Connects to a LiveKit room, starts the stream operator, and runs
a voice agent that can query feed content and control the stream.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from livekit.agents import (
    AgentSession,
    AgentServer,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
)
from livekit.plugins import silero

from cosmos_control.agent import CosmosAgent
from cosmos_control.operator import CVDisplayOperator
from cosmos_live.config import Config
from cosmos_storage import QdrantVectorStore

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


_setup_logging()

config = Config.from_yaml()
logger.info("Loaded config for room=%s", config.livekit.room)

vector_store = QdrantVectorStore(
    url=config.qdrant.url,
    collection_name=config.qdrant.collection_name,
    embedding_model=config.qdrant.embedding_model,
)

vector_store._ensure_model()

server = AgentServer(
    ws_url=config.livekit.url,
    api_key=config.livekit.api_key,
    api_secret=config.livekit.api_secret.get_secret_value(),
)

operator: CVDisplayOperator | None = None
if config.operator is not None:
    operator = CVDisplayOperator(config.operator, config.livekit)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    if operator is not None:
        async def start_operator():
            logger.info("Starting operator...")
            await operator.start()
            logger.info("Operator started successfully")

        asyncio.create_task(start_operator())

    agent = CosmosAgent(
        vector_store=vector_store,
        operator=operator,
        instructions=config.agent.instructions,
    )

    session = AgentSession(
        stt=inference.STT(config.agent.stt),
        llm=inference.LLM(config.agent.llm),
        tts=inference.TTS(config.agent.tts),
        vad=ctx.proc.userdata["vad"],
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    await session.generate_reply(
        instructions="Greet the user briefly and let them know you can help monitor feeds and control the stream."
    )


if __name__ == "__main__":
    cli.run_app(server)
