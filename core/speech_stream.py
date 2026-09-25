"""
core/speech_stream.py
Streaming TTS shared by the browser WebSocket and telephony: ordered delivery,
concurrent synthesis.

LLM text → SpeechChunker → one TTS task per segment (started immediately,
concurrency bounded by the router) → `emit(text, speech)` strictly in order.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from config.logging_config import get_logger
from config.settings import Settings
from core.chunker import SpeechChunker
from core.metrics import TurnTimer
from tts.base import Speech

logger = get_logger("core.speech")

Emit = Callable[[str, Speech], Awaitable[None]]


class StreamingTTS:
    def __init__(self, tts, emit: Emit, language: str, timer: TurnTimer, settings: Settings) -> None:
        self.tts, self.emit, self.language, self.timer = tts, emit, language, timer
        self.chunker = SpeechChunker(settings.tts_first_chunk_min_chars,
                                     settings.tts_first_chunk_max_chars)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending: list[asyncio.Task] = []
        self._sender = asyncio.create_task(self._send_loop())

    def _dispatch(self, segments: list[str]) -> None:
        for text in segments:
            task = asyncio.create_task(self.tts.synthesize(text, self.language))
            self._pending.append(task)
            self._queue.put_nowait((text, task))

    def feed(self, text: str) -> None:
        self._dispatch(self.chunker.feed(text))

    async def finish(self) -> None:
        self._dispatch(self.chunker.flush())
        self._queue.put_nowait(None)
        await self._sender

    def cancel(self) -> None:
        for task in (*self._pending, self._sender):
            task.cancel()

    async def _send_loop(self) -> None:
        while (item := await self._queue.get()) is not None:
            text, task = item
            try:
                speech = await task
            except Exception as exc:
                logger.error("TTS failed for %.40s: %r", text, exc)
                continue
            if speech is None:
                continue
            self.timer.mark("first_audio")
            await self.emit(text, speech)
