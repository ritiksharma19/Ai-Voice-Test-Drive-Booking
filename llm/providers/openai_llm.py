"""OpenAI backend (official SDK, Chat Completions streaming).

Also works with any OpenAI-compatible endpoint via OPENAI_BASE_URL
(Azure OpenAI v1 API, vLLM, LM Studio, …).
"""
from __future__ import annotations

from typing import AsyncIterator

from config.logging_config import get_logger
from config.settings import Settings
from llm.base import LLMBackend

logger = get_logger("llm.openai")


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, s: Settings) -> None:
        self.model = s.openai_model
        self._api_key = s.openai_api_key
        self._base_url = s.openai_base_url or None
        self._max_tokens = s.llm_max_tokens
        self._timeout = s.llm_request_timeout
        self._retries = s.llm_max_retries
        # "none" gives the lowest time-to-first-token on GPT-6 / GPT-5.x reasoning models.
        self._reasoning_effort = s.openai_reasoning_effort or None
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._api_key or self._base_url)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore
            self._client = AsyncOpenAI(
                api_key=self._api_key or "not-needed",
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._retries,
            )
        return self._client

    async def _create(self, messages: list[dict]):
        kwargs = dict(model=self.model, messages=messages, stream=True,
                      max_completion_tokens=self._max_tokens)
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        return await self._get_client().chat.completions.create(**kwargs)

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        from openai import BadRequestError  # type: ignore
        payload = [{"role": "system", "content": system}, *messages]
        try:
            stream = await self._create(payload)
        except BadRequestError as exc:
            if self._reasoning_effort and "reasoning" in str(exc).lower():
                logger.warning("%s rejected reasoning_effort=%s — retrying without it",
                               self.model, self._reasoning_effort)
                self._reasoning_effort = None
                stream = await self._create(payload)
            else:
                raise
        async for chunk in stream:
            if chunk.choices:
                text = chunk.choices[0].delta.content
                if text:
                    yield text

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
