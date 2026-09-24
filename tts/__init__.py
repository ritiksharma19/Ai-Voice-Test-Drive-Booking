"""
tts/__init__.py
TTSRouter — picks a voice engine per language with automatic fallback.

TTS_PROVIDER:
  auto        — Sarvam for Indian languages when SARVAM_API_KEY is set,
                Edge for English and whenever Sarvam is unavailable   [default]
  sarvam | openai | elevenlabs | edge — that engine first, Edge as fallback

Also:
  • Bounded concurrency (TTS_MAX_CONCURRENCY) so a long answer cannot burst
    past provider rate limits.
  • Per-attempt timeout (TTS_TIMEOUT) and a short cool-down for an engine
    that just failed, so the next sentence goes straight to the fallback.
  • Small LRU cache: repeated phrases (greetings, confirmations) cost 0 ms.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from tts.base import Speech, TTSBase

logger = get_logger("tts")

_COOLDOWN_S = 30.0


def _build(name: str, s: Settings) -> TTSBase:
    if name == "edge":
        from tts.edge_tts_engine import EdgeTTS
        return EdgeTTS()
    if name == "sarvam":
        from tts.sarvam import SarvamTTS
        return SarvamTTS(s)
    if name == "openai":
        from tts.cloud_tts import OpenAITTS
        return OpenAITTS(s)
    if name == "elevenlabs":
        from tts.cloud_tts import ElevenLabsTTS
        return ElevenLabsTTS(s)
    raise ValueError(f"Unknown TTS_PROVIDER '{name}' (auto | sarvam | openai | elevenlabs | edge)")


class TTSRouter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.mode = self.s.tts_provider
        names = ["sarvam", "edge"] if self.mode == "auto" else [self.mode, "edge"]
        self.engines: dict[str, TTSBase] = {}
        for name in dict.fromkeys(names):
            engine = _build(name, self.s)
            if engine.available:
                self.engines[name] = engine
            elif self.mode != "auto":
                logger.warning("TTS '%s' has no API key — falling back to Edge", name)
        self._sem = asyncio.Semaphore(self.s.tts_max_concurrency)
        self._cooldown_until: dict[str, float] = {}
        self._cache: OrderedDict[tuple[str, str], Speech] = OrderedDict()
        logger.info("TTS engines: %s (mode=%s)", ", ".join(self.engines), self.mode)

    def chain_for(self, language: str) -> list[TTSBase]:
        engines = [e for e in self.engines.values() if e.supports(language)]
        if self.mode == "auto" and language == "en":
            engines.sort(key=lambda e: e.name != "edge")   # English → Edge first
        return engines or [self.engines["edge"]]

    def describe(self) -> str:
        return "+".join(self.engines)

    async def synthesize(self, text: str, language: str = "en") -> Speech | None:
        key = (language, text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        chain = self.chain_for(language)
        now = time.monotonic()
        chain = ([e for e in chain if self._cooldown_until.get(e.name, 0) <= now]
                 + [e for e in chain if self._cooldown_until.get(e.name, 0) > now])
        async with self._sem:
            for engine in chain:
                t0 = time.perf_counter()
                try:
                    speech = await asyncio.wait_for(engine.synthesize(text, language),
                                                    timeout=self.s.tts_timeout)
                except asyncio.TimeoutError:
                    speech = None
                    logger.warning("TTS %s timed out after %.1fs", engine.name, self.s.tts_timeout)
                if speech is not None:
                    self._cooldown_until.pop(engine.name, None)
                    logger.debug("TTS %s %.0f ms | %.50s", engine.name,
                                 (time.perf_counter() - t0) * 1000, text)
                    self._remember(key, speech)
                    return speech
                self._cooldown_until[engine.name] = time.monotonic() + _COOLDOWN_S
        logger.error("All TTS engines failed for: %.60s", text)
        return None

    def _remember(self, key: tuple[str, str], speech: Speech) -> None:
        if self.s.tts_cache_size <= 0 or len(key[1]) > 200:
            return
        self._cache[key] = speech
        while len(self._cache) > self.s.tts_cache_size:
            self._cache.popitem(last=False)

    async def warmup(self) -> None:
        await asyncio.gather(*(e.warmup() for e in self.engines.values()), return_exceptions=True)

    async def close(self) -> None:
        await asyncio.gather(*(e.close() for e in self.engines.values()), return_exceptions=True)


def build_tts_router(settings: Settings | None = None) -> TTSRouter:
    return TTSRouter(settings)


__all__ = ["TTSRouter", "build_tts_router", "Speech", "TTSBase"]
