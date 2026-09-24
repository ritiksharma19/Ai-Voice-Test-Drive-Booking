"""LLM backend registry. Add a provider by implementing llm.base.LLMBackend."""
from __future__ import annotations

from config.settings import Settings
from llm.base import LLMBackend


def build_backend(name: str, settings: Settings) -> LLMBackend:
    if name == "gemini":
        from llm.providers.gemini import GeminiBackend
        return GeminiBackend(settings)
    if name == "openai":
        from llm.providers.openai_llm import OpenAIBackend
        return OpenAIBackend(settings)
    if name in ("anthropic", "claude"):
        from llm.providers.anthropic_llm import AnthropicBackend
        return AnthropicBackend(settings)
    if name == "ollama":
        from llm.providers.ollama_llm import OllamaBackend
        return OllamaBackend(settings)
    raise ValueError(f"Unknown LLM provider '{name}' (use gemini | openai | anthropic | ollama)")


__all__ = ["build_backend"]
