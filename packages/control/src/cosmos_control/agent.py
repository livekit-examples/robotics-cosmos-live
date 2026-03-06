from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from livekit.agents import Agent, RunContext, function_tool

from cosmos_core import Overlay

if TYPE_CHECKING:
    from cosmos_control.operator import CVDisplayOperator
    from cosmos_storage.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


class CosmosAgent(Agent):
    """LiveKit voice agent that queries feed content and controls the stream."""

    def __init__(
        self,
        *,
        vector_store: QdrantVectorStore,
        operator: CVDisplayOperator | None = None,
        instructions: str = "",
    ) -> None:
        super().__init__(instructions=instructions)
        self._vector_store = vector_store
        self._operator = operator
        self._monitor_task: asyncio.Task[None] | None = None
        self._monitor_query: str | None = None

    # ------------------------------------------------------------------
    # Utility tools
    # ------------------------------------------------------------------

    @function_tool
    async def get_current_time(self, ctx: RunContext) -> str:
        """Get the current time as a unix timestamp in seconds (e.g. 1770628192.75). All timestamps in this system use unix seconds. Call this before using the `since` parameter in search so you can compute the correct cutoff."""
        return str(time.time())

    # ------------------------------------------------------------------
    # Search tools
    # ------------------------------------------------------------------

    @function_tool
    async def query(
        self,
        ctx: RunContext,
        text: str,
        top_k: int = 5,
    ) -> str:
        """Semantic search across all feed content. Use when you need to find feeds by meaning (e.g. "the feed with a dog", "someone cooking").

        Args:
            text: Natural language query to search for.
            top_k: Maximum number of results to return.
        """
        docs = await self._vector_store.query(text=text, top_k=top_k)
        if not docs:
            return "No results found."
        lines: list[str] = []
        for doc in docs:
            age_mins = (time.time() - doc.timestamp) / 60
            lines.append(
                f"[{doc.track_id} | {age_mins:.0f}m ago] {doc.content}"
            )
        return "\n".join(lines)

    @function_tool
    async def search(
        self,
        ctx: RunContext,
        kind: str | None = None,
        track_id: str | None = None,
        since: float | None = None,
        top_k: int = 10,
    ) -> str:
        """Filter-based search on stored documents. No semantic ranking; results ordered by most recent first.

        Use this to retrieve description on the audio or video content of specific feeds, or within a certain time range.
        All filter parameters are optional — omit any you don't need.
        
        Any time results should be spoken after converting to pacific time like two thirty pm, or five thirty six am.

        Args:
            kind: either "video" or "audio" is availbale. you should use "video" to get video frame analysis and "audio" to get audio transcription.
            track_id: Filter by feed/participant identity, e.g. "cam-1".
            since: Only return documents with timestamp greater than this value (unix seconds, e.g. 1770628192.75). Call get_current_time first, then subtract to get the cutoff (e.g. current_time - 300 for the last 5 minutes).
            top_k: Maximum number of results to return.
        """
        docs = await self._vector_store.search(kind=kind, track_id=track_id, since=since, top_k=top_k)
        if not docs:
            return "No results found."
        lines: list[str] = []
        for doc in docs:
            age_mins = (time.time() - doc.timestamp) / 60
            lines.append(
                f"[{doc.track_id} | {age_mins:.0f}m ago] {doc.content}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Feed monitoring tools
    # ------------------------------------------------------------------

    @function_tool
    async def monitor_feed(self, ctx: RunContext, query: str) -> str:
        """Start monitoring all feeds for content matching a semantic query.
        Runs a background search every 2 seconds and automatically switches
        the active feed when new matching content is detected.

        If a monitor is already active, it is replaced with the new query.

        Args:
            query: Natural language description of what to watch for (e.g. "a person waving", "a dog playing").
        """
        await self._cancel_monitor()
        self._monitor_query = query
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        return f"Now monitoring feeds for: '{query}'. Will auto-switch when a match is detected."

    @function_tool
    async def stop_monitoring_feed(self, ctx: RunContext) -> str:
        """Stop the active feed monitor that was started with monitor_feed."""
        if self._monitor_task is None:
            return "No active feed monitor."
        query = self._monitor_query
        await self._cancel_monitor()
        return f"Stopped monitoring for: '{query}'"

    async def _cancel_monitor(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)
            self._monitor_task = None
        self._monitor_query = None

    _MONITOR_SCORE_THRESHOLD = 0.25

    async def _monitor_loop(self) -> None:
        """Background loop: semantic query every 1s, auto-switch on recent hit."""
        try:
            while True:
                await asyncio.sleep(1)
                if self._monitor_query is None:
                    break

                now = time.time()
                since = now - 10
                docs = await self._vector_store.query(
                    text=self._monitor_query,
                    top_k=10,
                    since=since,
                    score_threshold=self._MONITOR_SCORE_THRESHOLD,
                )

                current_feed = (
                    self._operator.active_feed
                    if self._operator is not None
                    else None
                )
                candidates = [
                    d for d in docs if d.track_id != current_feed
                ] if current_feed else docs

                if docs:
                    logger.info(
                        "Monitor query %r: %d results above threshold %.2f "
                        "(%d on other feeds, active=%r)",
                        self._monitor_query,
                        len(docs),
                        self._MONITOR_SCORE_THRESHOLD,
                        len(candidates),
                        current_feed,
                    )
                else:
                    logger.debug(
                        "Monitor query %r: no results above threshold %.2f",
                        self._monitor_query,
                        self._MONITOR_SCORE_THRESHOLD,
                    )

                if not candidates:
                    continue

                best = max(candidates, key=lambda d: d.score)
                logger.info(
                    "Monitor hit for %r — switching feed=%s score=%.3f content=%s",
                    self._monitor_query,
                    best.track_id,
                    best.score,
                    best.content[:120],
                )

                if self._operator is not None:
                    await self._operator.set_feed(best.track_id)

                try:
                    session = self.session
                    await session.say(f"Switching to {best.track_id}")
                    # session = self.session
                    # await session.generate_reply(
                    #     instructions=(
                    #         f"The feed monitor detected a match for '{self._monitor_query}' "
                    #         f"on feed '{best.track_id}'. Matching content: {best.content[:200]}. "
                    #         f"The feed has been automatically switched. Briefly inform the user."
                    #     )
                    # )
                except Exception:
                    logger.debug(
                        "Could not notify session of monitor hit", exc_info=True
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Feed monitor loop error")

    # ------------------------------------------------------------------
    # Stream control tools
    # ------------------------------------------------------------------

    @function_tool
    async def switch_feed(self, ctx: RunContext, feed_id: str) -> str:
        """Switch the active stream to a different feed.

        Args:
            feed_id: The feed participant identity to switch to.
        """
        if self._operator is None:
            return "Stream operator is not configured."
        await self._operator.set_feed(feed_id)
        return f"Switched to feed {feed_id}."

    @function_tool
    async def list_feeds(self, ctx: RunContext) -> str:
        """List all available feed participant identities."""
        if self._operator is None:
            return "Stream operator is not configured."
        feeds = self._operator.available_feeds
        if not feeds:
            return "No feeds currently available."
        return "Available feeds: " + ", ".join(feeds)

    @function_tool
    async def set_overlay(
        self, ctx: RunContext, slot: str, text: str
    ) -> str:
        """Set a text overlay on the stream output.

        Args:
            slot: Overlay slot name — one of 'lower_third', 'title', or 'banner'.
            text: The text to display.
        """
        if self._operator is None:
            return "Stream operator is not configured."
        overlay = Overlay(kind="text", data={"text": text})
        await self._operator.set_overlay(slot, overlay)
        return f"Overlay set on slot '{slot}'."

    @function_tool
    async def clear_overlay(self, ctx: RunContext, slot: str) -> str:
        """Remove an overlay from the stream.

        Args:
            slot: Overlay slot name to clear.
        """
        if self._operator is None:
            return "Stream operator is not configured."
        await self._operator.clear_overlay(slot)
        return f"Overlay cleared from slot '{slot}'."

    @function_tool
    async def start_stream(self, ctx: RunContext) -> str:
        """Start the stream operator."""
        if self._operator is None:
            return "Stream operator is not configured."
        await self._operator.start()
        return "Stream started."

    @function_tool
    async def stop_stream(self, ctx: RunContext) -> str:
        """Stop the stream operator."""
        if self._operator is None:
            return "Stream operator is not configured."
        await self._operator.stop()
        return "Stream stopped."
