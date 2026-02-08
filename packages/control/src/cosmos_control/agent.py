from __future__ import annotations

from cosmos_core import VectorStore, StreamOperator


class ControlAgent:
    """LLM-powered agent that queries RAG stores and controls the stream."""

    def __init__(
        self,
        vector_store: VectorStore,
        stream_operator: StreamOperator,
    ) -> None:
        self._vector_store = vector_store
        self._stream_operator = stream_operator

    async def handle(self, user_prompt: str) -> str:
        """Process a single user prompt: query stores, reason, and act."""
        raise NotImplementedError

    async def run(self) -> None:
        """Main control loop — listen for prompts and handle them."""
        raise NotImplementedError
