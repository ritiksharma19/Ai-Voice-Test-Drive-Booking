"""
stt/base.py
Interface for speech-to-text backends.

`transcribe(samples, sample_rate, language)` is async for every backend:
cloud engines await HTTP natively; local GPU engines run their blocking
inference in a worker thread behind a lock, so the event loop never blocks
and the GPU processes one utterance at a time (queued requests wait in order).
"""
from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod

import numpy as np


class STTBase(ABC):
    name: str = "stt"

    @abstractmethod
    async def transcribe(self, samples: np.ndarray, sample_rate: int = 16000,
                         language: str | None = None) -> dict:
        """Return {"text": str, "language": ISO-639-1 code}."""

    def describe(self) -> str:
        return self.name

    async def close(self) -> None:
        return None


class LocalSTT(STTBase):
    """Base for in-process models: blocking inference off the event loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @abstractmethod
    def _transcribe_sync(self, samples: np.ndarray, language: str | None) -> dict:
        ...

    async def transcribe(self, samples: np.ndarray, sample_rate: int = 16000,
                         language: str | None = None) -> dict:
        from stt.audio import to_16k
        audio = to_16k(samples, sample_rate)

        def run() -> dict:
            with self._lock:
                return self._transcribe_sync(audio, language)
        return await asyncio.to_thread(run)
