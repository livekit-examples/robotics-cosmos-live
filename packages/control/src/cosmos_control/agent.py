from __future__ import annotations

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
