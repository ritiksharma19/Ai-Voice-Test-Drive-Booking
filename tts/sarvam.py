"""
tts/sarvam.py
Sarvam AI Bulbul v3 — the most natural voices for Indian languages
(handles code-mixed Hinglish and number normalisation natively).
Requests MP3 so payloads are ~10x smaller than WAV.
"""
from __future__ import annotations

from config.logging_config import get_logger
from config.settings import Settings
from core.http import get_http_client, request_with_retry
from tts.base import Speech, TTSBase

logger = get_logger("tts.sarvam")

_URL = "https://api.sarvam.ai/text-to-speech"
_LANG = {
    "en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "bn": "bn-IN", "gu": "gu-IN", "mr": "mr-IN", "pa": "pa-IN",
}


class SarvamTTS(TTSBase):
    name = "sarvam"

    def __init__(self, s: Settings) -> None:
        self._key = s.sarvam_api_key
        self._model = s.sarvam_tts_model
        self._speaker = s.sarvam_speaker
        self._timeout = s.tts_timeout

    @property
    def available(self) -> bool:
        return bool(self._key)

    def supports(self, language: str) -> bool:
        return language in _LANG

    async def synthesize(self, text: str, language: str = "en") -> Speech | None:
        payload = {
            "text": text,
            "language_code": _LANG.get(language, "en-IN"),
            "speaker": self._speaker,
            "model": self._model,
            "output_audio_codec": "mp3",
            "speech_sample_rate": 24000,
        }
        client = get_http_client()
        try:
            resp = await request_with_retry(
                lambda: client.post(_URL, json=payload, timeout=self._timeout,
                                    headers={"api-subscription-key": self._key}),
                label="sarvam-tts")
            if resp.status_code != 200:
                logger.warning("Sarvam TTS HTTP %d: %s", resp.status_code, resp.text[:200])
                return None
            audios = resp.json().get("audios") or []
        except Exception as exc:
            logger.warning("Sarvam TTS error: %r", exc)
            return None
        return Speech("".join(audios), "mp3") if audios else None

    async def warmup(self) -> None:
        # Opens the TLS connection so the first real sentence skips the handshake.
        try:
            await get_http_client().head("https://api.sarvam.ai", timeout=3)
        except Exception:
            pass
