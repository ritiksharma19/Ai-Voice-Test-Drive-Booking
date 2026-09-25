"""
llm/base.py
Interface every LLM backend implements.

Backends are stateless text streamers: the orchestrator owns prompts,
history, retrieval and failover. Messages use the neutral format
    [{"role": "user" | "assistant", "content": str}, ...]
and each backend converts to its SDK's wire format.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMBackend(ABC):
    name: str = "base"
    model: str = ""

    @property
    def available(self) -> bool:
        """False when required credentials are missing (backend is skipped)."""
        return True

    @abstractmethod
    def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        """Yield text deltas as soon as the provider produces them."""

    async def warmup(self, system: str = "Reply with one word.") -> None:
        """Open connections / load weights so the first user turn is fast. Pass the
        real system prompt to also fill prompt caches (Ollama's KV cache, OpenAI's
        prefix cache), so the first turn skips re-reading it."""
        async for _ in self.stream(system, [{"role": "user", "content": "Hi"}]):
            break

    async def close(self) -> None:
        return None

    def describe(self) -> str:
        return f"{self.name}:{self.model}"
