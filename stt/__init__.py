"""
stt/__init__.py
STT factory. STT_PROVIDER selects the backend:
  whisper  — faster-whisper on the local NVIDIA GPU (CPU fallback)   [default]
  seamless — AI4Bharat Indic-Seamless on the local GPU (optional extras)
  sarvam   — Sarvam AI Saaras (cloud, best for Indian languages)
  openai   — OpenAI transcription (cloud)
"""
from __future__ import annotations

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from stt.base import STTBase

logger = get_logger("stt")


def build_stt_engine(settings: Settings | None = None) -> STTBase:
    s = settings or get_settings()
    provider = s.stt_provider
    if provider in ("whisper", "faster-whisper", "faster_whisper"):
        from stt.faster_whisper_stt import FasterWhisperSTT
        engine: STTBase = FasterWhisperSTT(s)
    elif provider == "seamless":
        from stt.seamless import SeamlessSTT
        engine = SeamlessSTT(s)
    elif provider == "sarvam":
        from stt.cloud_stt import SarvamSTT
        engine = SarvamSTT(s)
    elif provider == "openai":
        from stt.cloud_stt import OpenAISTT
        engine = OpenAISTT(s)
    else:
        raise ValueError(f"Unknown STT_PROVIDER '{provider}' (whisper | seamless | sarvam | openai)")
    logger.info("STT engine: %s", engine.describe())
    return engine


__all__ = ["build_stt_engine", "STTBase"]
