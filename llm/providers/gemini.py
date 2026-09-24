"""Google Gemini backend (google-genai SDK, async streaming)."""
from __future__ import annotations

from typing import AsyncIterator

from config.logging_config import get_logger
from config.settings import Settings
from llm.base import LLMBackend

logger = get_logger("llm.gemini")


class GeminiBackend(LLMBackend):
    name = "gemini"

    def __init__(self, s: Settings) -> None:
        self.model = s.gemini_model
        self._api_key = s.gemini_api_key
        self._max_tokens = s.llm_max_tokens
        self._timeout_ms = int(s.llm_request_timeout * 1000)
        self._retries = s.llm_max_retries
        # Gemini 3.x uses thinking levels; "minimal" is the lowest-latency setting
        # (Flash-Lite). Models that reject the level fall back to their default.
        self._thinking_level = s.gemini_thinking_level or None
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=self._timeout_ms,
                    retry_options=types.HttpRetryOptions(
                        attempts=self._retries + 1, initial_delay=0.25, max_delay=2.0),
                ),
            )
        return self._client

    def _config(self, system: str):
        from google.genai import types  # type: ignore
        kwargs = dict(system_instruction=system, max_output_tokens=self._max_tokens)
        if self._thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self._thinking_level)
        return types.GenerateContentConfig(**kwargs)

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        client = self._get_client()
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        try:
            response = await client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=self._config(system))
        except Exception as exc:
            if self._thinking_level and "thinking" in str(exc).lower():
                logger.warning("%s rejected thinking_level=%s — using model default",
                               self.model, self._thinking_level)
                self._thinking_level = None
                response = await client.aio.models.generate_content_stream(
                    model=self.model, contents=contents, config=self._config(system))
            else:
                raise
        async for chunk in response:
            text = chunk.text
            if text:
                yield text

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aio.aclose()
            except Exception:
                pass
