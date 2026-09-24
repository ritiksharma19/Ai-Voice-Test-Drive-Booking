"""
tts/base.py
Interface for text-to-speech backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Speech:
    audio_b64: str   # base64-encoded audio
    fmt: str         # "mp3" | "wav" — tells the browser which MIME type to use


class TTSBase(ABC):
    name: str = "tts"

    @property
    def available(self) -> bool:
        return True

    def supports(self, language: str) -> bool:
        return True

    @abstractmethod
    async def synthesize(self, text: str, language: str = "en") -> Speech | None:
        """Return synthesized speech, or None on failure (router falls back)."""

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None
