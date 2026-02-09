from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from livekit.agents import Agent, RunContext, function_tool

from cosmos_core import Overlay

if TYPE_CHECKING:
    from cosmos_control.operator import CVDisplayOperator
    from cosmos_storage.milvus import MilvusVectorStore

logger = logging.getLogger(__name__)


class CosmosAgent(Agent):
    """LiveKit voice agent that queries feed content and controls the stream."""

    def __init__(
        self,
        *,
        vector_store: MilvusVectorStore,
        operator: CVDisplayOperator | None = None,
        instructions: str = "",
    ) -> None:
        super().__init__(instructions=instructions)
        self._vector_store = vector_store
        self._operator = operator

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
        expr: str = "",
        top_k: int = 10,
    ) -> str:
        """Filter-based search on document fields — like a SQL WHERE clause. No semantic ranking; results ordered by timestamp descending.

        Available fields: track_id (str), kind (str), timestamp (float), content (str).

        Example filters:
          'track_id == "cam-1"'
          'timestamp > 1700000000'
          'track_id == "cam-1" and timestamp > 1700000000'
          'kind == "analysis"'
          'track_id in ["cam-1", "cam-2"]'

        Args:
            expr: Milvus filter expression. Empty string returns all documents.
            top_k: Maximum number of results to return.
        """
        docs = await self._vector_store.search(expr=expr, top_k=top_k)
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
