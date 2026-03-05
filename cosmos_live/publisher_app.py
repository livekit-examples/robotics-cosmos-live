"""Entry point for publishing a local camera to a LiveKit room.

Captures video from a local device (e.g. webcam) and publishes it
as a video track in the configured LiveKit room.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from cosmos_live.config import Config
from cosmos_ingestion.publisher import Publisher

logger = logging.getLogger("cosmos_live.publisher")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a local camera to a LiveKit room")
    parser.add_argument(
        "--identity",
        default="publisher",
        help="Identity used when connecting to LiveKit (default: publisher)",
    )
    parser.add_argument(
        "--camera-index",
        default="0",
        help="Camera device index or URL passed to the video source (default: 0)",
    )
    return parser.parse_args()


async def main() -> None:
    _setup_logging()
    args = _parse_args()

    config = Config.from_yaml()
    logger.info("Loaded config for room=%s", config.livekit.room)

    publisher = Publisher(
        config=config.livekit,
        source=args.camera_index,
        fps=config.operator.fps if config.operator else 30,
        width=config.operator.width if config.operator else 1280,
        height=config.operator.height if config.operator else 720,
        identity=args.identity,
    )

    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    publisher_task = asyncio.create_task(publisher.run())

    logger.info("Publisher started — press Ctrl+C to stop")

    done, _ = await asyncio.wait(
        [publisher_task, asyncio.ensure_future(stop)],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if publisher_task.done() and publisher_task.exception():
        logger.error("Publisher failed", exc_info=publisher_task.exception())

    logger.info("Shutting down…")

    publisher.stop()
    if not publisher_task.done():
        publisher_task.cancel()
        try:
            await publisher_task
        except asyncio.CancelledError:
            pass

    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
