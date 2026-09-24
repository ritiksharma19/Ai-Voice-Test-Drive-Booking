"""
tts/edge_tts_engine.py
Microsoft Edge neural voices via the free `edge-tts` service (no API key;
needs internet access). Always available, so it is the universal fallback.
"""
from __future__ import annotations

import base64

import edge_tts  # type: ignore

from config.logging_config import get_logger
from tts.base import Speech, TTSBase

logger = get_logger("tts.edge")

_VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "gu": "gu-IN-DhwaniNeural",
    "mr": "mr-IN-AarohiNeural",
    "pa": "pa-IN-OjasNeural",
}


class EdgeTTS(TTSBase):
    name = "edge"

    async def synthesize(self, text: str, language: str = "en") -> Speech | None:
        voice = _VOICES.get(language, _VOICES["en"])
        chunks: list[bytes] = []
        try:
            async for chunk in edge_tts.Communicate(text, voice).stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
        except Exception as exc:
            logger.warning("Edge TTS error (%s): %r", voice, exc)
            return None
        if not chunks:
            return None
        return Speech(base64.b64encode(b"".join(chunks)).decode("ascii"), "mp3")
