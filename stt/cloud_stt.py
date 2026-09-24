"""
stt/cloud_stt.py
Cloud speech-to-text backends (no local GPU required).

  • SarvamSTT — Sarvam AI Saaras v3: best accuracy for Indian languages and
    code-mixed speech (Hinglish), auto language detection.
  • OpenAISTT — OpenAI gpt-4o-mini-transcribe (or gpt-transcribe): strong
    multilingual accuracy, simple REST.
Both upload the utterance as 16 kHz WAV over the shared keep-alive client.
"""
from __future__ import annotations

import time

import numpy as np

from config.logging_config import get_logger
from config.settings import Settings
from core.http import get_http_client, request_with_retry
from core.lang import detect_language, normalize_language
from stt.audio import encode_wav, is_hallucination, to_16k, trim_silence
from stt.base import STTBase

logger = get_logger("stt.cloud")


def _prepare(samples: np.ndarray, sample_rate: int) -> bytes | None:
    audio = trim_silence(to_16k(samples, sample_rate))
    return encode_wav(audio) if audio.size >= 1600 else None


class SarvamSTT(STTBase):
    name = "sarvam"
    _URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, s: Settings) -> None:
        if not s.sarvam_api_key:
            raise ValueError("STT_PROVIDER=sarvam requires SARVAM_API_KEY")
        self._key = s.sarvam_api_key
        self._model = s.sarvam_stt_model
        self._forced = s.stt_language or None
        self._timeout = 10.0

    def describe(self) -> str:
        return f"sarvam:{self._model}"

    async def transcribe(self, samples, sample_rate=16000, language=None) -> dict:
        t0 = time.perf_counter()
        wav = _prepare(samples, sample_rate)
        lang_hint = language or self._forced
        if wav is None:
            return {"text": "", "language": lang_hint or "en"}
        data = {"model": self._model, "mode": "transcribe"}
        if lang_hint:
            data["language_code"] = f"{lang_hint}-IN"
        client = get_http_client()
        resp = await request_with_retry(
            lambda: client.post(self._URL, headers={"api-subscription-key": self._key},
                                data=data, files={"file": ("audio.wav", wav, "audio/wav")},
                                timeout=self._timeout),
            label="sarvam-stt")
        resp.raise_for_status()
        body = resp.json()
        text = (body.get("transcript") or "").strip()
        lang = normalize_language(body.get("language_code"), default="") or detect_language(text, lang_hint)
        if is_hallucination(text):
            text = ""
        logger.info("STT %.0f ms | %s | %r", (time.perf_counter() - t0) * 1000, lang, text[:80])
        return {"text": text, "language": lang}


class OpenAISTT(STTBase):
    name = "openai"

    def __init__(self, s: Settings) -> None:
        if not (s.openai_api_key or s.openai_base_url):
            raise ValueError("STT_PROVIDER=openai requires OPENAI_API_KEY")
        from openai import AsyncOpenAI  # type: ignore
        self._client = AsyncOpenAI(api_key=s.openai_api_key or "not-needed",
                                   base_url=s.openai_base_url or None,
                                   timeout=10.0, max_retries=1)
        self._model = s.openai_stt_model
        self._forced = s.stt_language or None

    def describe(self) -> str:
        return f"openai:{self._model}"

    async def transcribe(self, samples, sample_rate=16000, language=None) -> dict:
        t0 = time.perf_counter()
        wav = _prepare(samples, sample_rate)
        lang_hint = language or self._forced
        if wav is None:
            return {"text": "", "language": lang_hint or "en"}
        kwargs = {"model": self._model, "file": ("audio.wav", wav, "audio/wav")}
        if lang_hint:
            kwargs["language"] = lang_hint
        result = await self._client.audio.transcriptions.create(**kwargs)
        text = (getattr(result, "text", "") or "").strip()
        lang = detect_language(text, lang_hint)
        if is_hallucination(text):
            text = ""
        logger.info("STT %.0f ms | %s | %r", (time.perf_counter() - t0) * 1000, lang, text[:80])
        return {"text": text, "language": lang}

    async def close(self) -> None:
        await self._client.close()
