"""Ollama backend — local models on the NVIDIA GPU, fully offline."""
from __future__ import annotations

from typing import AsyncIterator

from config.logging_config import get_logger
from config.settings import Settings
from llm.base import LLMBackend

logger = get_logger("llm.ollama")


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, s: Settings) -> None:
        self.model = s.ollama_model
        self._host = s.ollama_host
        self._keep_alive = s.ollama_keep_alive
        self._options = {"num_predict": s.llm_max_tokens, "temperature": s.llm_temperature}
        self._think = {"true": True, "false": False}.get(s.ollama_think)
        self._timeout = s.llm_request_timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama  # type: ignore
            self._client = ollama.AsyncClient(host=self._host, timeout=self._timeout)
        return self._client

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            stream=True,
            options=self._options,
            # Keeps weights resident in VRAM between turns — avoids multi-second reloads.
            keep_alive=self._keep_alive,
        )
        if self._think is not None:
            kwargs["think"] = self._think
        async for chunk in await self._get_client().chat(**kwargs):
            text = chunk["message"]["content"]
            if text:
                yield text
