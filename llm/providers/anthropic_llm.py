"""Anthropic Claude backend (official SDK, Messages API streaming)."""
from __future__ import annotations

from typing import AsyncIterator

from config.logging_config import get_logger
from config.settings import Settings
from llm.base import LLMBackend

logger = get_logger("llm.anthropic")


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, s: Settings) -> None:
        self.model = s.anthropic_model
        self._api_key = s.anthropic_api_key
        self._max_tokens = s.llm_max_tokens
        self._timeout = s.llm_request_timeout
        self._retries = s.llm_max_retries
        self._thinking = s.anthropic_thinking
        self._effort = s.anthropic_effort
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic  # type: ignore
            self._client = AsyncAnthropic(
                api_key=self._api_key, timeout=self._timeout, max_retries=self._retries)
        return self._client

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        kwargs: dict = dict(model=self.model, max_tokens=self._max_tokens,
                            system=system, messages=messages)
        # Haiku 4.5 runs without extended thinking by default (fastest TTFT).
        # Newer models think adaptively by default; ANTHROPIC_THINKING /
        # ANTHROPIC_EFFORT let you trade depth for latency per model.
        if self._thinking in ("disabled", "adaptive"):
            kwargs["thinking"] = {"type": self._thinking}
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        async with self._get_client().messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
