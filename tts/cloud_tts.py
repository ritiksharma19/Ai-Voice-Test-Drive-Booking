"""
tts/cloud_tts.py
Premium cloud voices:
  • OpenAITTS     — gpt-4o-mini-tts: expressive, steerable, multilingual.
  • ElevenLabsTTS — Flash v2.5: ~75 ms model latency, 32 languages incl. Hindi & Tamil.
"""
from __future__ import annotations

import base64

from config.logging_config import get_logger
from config.settings import Settings
from core.http import get_http_client, request_with_retry
from tts.base import Speech, TTSBase

logger = get_logger("tts.cloud")


class OpenAITTS(TTSBase):
    name = "openai"

    def __init__(self, s: Settings) -> None:
        self._key = s.openai_api_key
        self._base_url = s.openai_base_url or None
        self._model = s.openai_tts_model
        self._voice = s.openai_tts_voice
        self._timeout = s.tts_timeout
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._key or self._base_url)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore
            self._client = AsyncOpenAI(api_key=self._key or "not-needed", base_url=self._base_url,
                                       timeout=self._timeout, max_retries=1)
        return self._client

    async def synthesize(self, text: str, language: str = "en") -> Speech | None:
        try:
            resp = await self._get_client().audio.speech.create(
                model=self._model, voice=self._voice, input=text, response_format="mp3",
                instructions="Speak naturally and warmly, like a helpful assistant on a call.")
            audio = resp.content
        except Exception as exc:
            logger.warning("OpenAI TTS error: %r", exc)
            return None
        return Speech(base64.b64encode(audio).decode("ascii"), "mp3") if audio else None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


class ElevenLabsTTS(TTSBase):
    name = "elevenlabs"
    _DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"   # premade multilingual voice
    _LANGS = {"en", "hi", "ta"}              # Flash v2.5 Indic coverage

    def __init__(self, s: Settings) -> None:
        self._key = s.elevenlabs_api_key
        self._model = s.elevenlabs_model
        self._voice = s.elevenlabs_voice_id or self._DEFAULT_VOICE
        self._timeout = s.tts_timeout

    @property
    def available(self) -> bool:
        return bool(self._key)

    def supports(self, language: str) -> bool:
        return language in self._LANGS

    async def synthesize(self, text: str, language: str = "en") -> Speech | None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice}"
        body = {"text": text, "model_id": self._model, "language_code": language}
        client = get_http_client()
        try:
            resp = await request_with_retry(
                lambda: client.post(url, json=body, timeout=self._timeout,
                                    params={"output_format": "mp3_44100_64"},
                                    headers={"xi-api-key": self._key}),
                label="elevenlabs")
            if resp.status_code != 200:
                logger.warning("ElevenLabs HTTP %d: %s", resp.status_code, resp.text[:200])
                return None
        except Exception as exc:
            logger.warning("ElevenLabs error: %r", exc)
            return None
        return Speech(base64.b64encode(resp.content).decode("ascii"), "mp3") if resp.content else None
