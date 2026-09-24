"""
config/settings.py
Single source of truth for runtime configuration. Every value comes from the
environment (.env) — nothing provider-specific or secret is hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"').strip("'")


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, str(default)).lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str = "") -> list[str]:
    return [p.strip().lower() for p in _str(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: _str("LLM_PROVIDER", "gemini").lower())
    llm_fallbacks: list[str] = field(default_factory=lambda: _list("LLM_FALLBACKS", ""))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 400))
    llm_temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.6))
    llm_first_token_timeout: float = field(default_factory=lambda: _float("LLM_FIRST_TOKEN_TIMEOUT", 6.0))
    llm_request_timeout: float = field(default_factory=lambda: _float("LLM_REQUEST_TIMEOUT", 30.0))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 1))
    llm_max_concurrency: int = field(default_factory=lambda: _int("LLM_MAX_CONCURRENCY", 32))
    history_turns: int = field(default_factory=lambda: _int("HISTORY_TURNS", 8))
    history_chars: int = field(default_factory=lambda: _int("HISTORY_CHARS", 6000))

    gemini_api_key: str = field(default_factory=lambda: _str("GEMINI_API_KEY") or _str("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _str("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    gemini_thinking_level: str = field(default_factory=lambda: _str("GEMINI_THINKING_LEVEL", "minimal").lower())

    openai_api_key: str = field(default_factory=lambda: _str("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _str("OPENAI_BASE_URL"))
    openai_model: str = field(default_factory=lambda: _str("OPENAI_MODEL", "gpt-6-luna"))
    openai_reasoning_effort: str = field(default_factory=lambda: _str("OPENAI_REASONING_EFFORT", "none").lower())

    anthropic_api_key: str = field(default_factory=lambda: _str("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _str("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    # "" = model default | "disabled" | "adaptive"  (see README → Anthropic)
    anthropic_thinking: str = field(default_factory=lambda: _str("ANTHROPIC_THINKING", "").lower())
    # "" = model default | low | medium | high  (not supported on Haiku 4.5)
    anthropic_effort: str = field(default_factory=lambda: _str("ANTHROPIC_EFFORT", "").lower())

    ollama_host: str = field(default_factory=lambda: _str("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _str("OLLAMA_MODEL", "gemma3:12b"))
    # "" = model default; "false" disables reasoning on thinking models (qwen3, gpt-oss …)
    ollama_think: str = field(default_factory=lambda: _str("OLLAMA_THINK", "").lower())
    ollama_keep_alive: str = field(default_factory=lambda: _str("OLLAMA_KEEP_ALIVE", "30m"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_mode: str = field(default_factory=lambda: _str("RETRIEVAL_MODE", "auto").lower())
    retrieval_wait: float = field(default_factory=lambda: _float("LLM_RETRIEVAL_WAIT", 1.2))
    retrieval_timeout: float = field(default_factory=lambda: _float("RETRIEVAL_TIMEOUT", 4.0))
    retrieval_cache_ttl: int = field(default_factory=lambda: _int("RETRIEVAL_CACHE_TTL", 300))
    web_search_enabled: bool = field(default_factory=lambda: _bool("WEB_SEARCH_ENABLED", True))
    brave_search_api_key: str = field(default_factory=lambda: _str("BRAVE_SEARCH_API_KEY"))
    gcp_project_id: str = field(default_factory=lambda: _str("GCP_PROJECT_ID"))
    gcp_location: str = field(default_factory=lambda: _str("GCP_LOCATION", "global"))
    gcp_data_store_id: str = field(default_factory=lambda: _str("GCP_DATA_STORE_ID"))
    kb_multilingual: bool = field(default_factory=lambda: _bool("KB_MULTILINGUAL", False))

    # ── STT ──────────────────────────────────────────────────────────────────
    stt_provider: str = field(default_factory=lambda: _str("STT_PROVIDER", "whisper").lower())
    stt_language: str = field(default_factory=lambda: _str("STT_LANGUAGE", "").lower())
    stt_languages: list[str] = field(default_factory=lambda: _list(
        "STT_LANGUAGES", "en,hi,ta,te,kn,ml,bn,mr,gu,pa"))
    whisper_model: str = field(default_factory=lambda: _str("WHISPER_MODEL", ""))
    whisper_device: str = field(default_factory=lambda: _str("WHISPER_DEVICE", "auto").lower())
    whisper_compute_type: str = field(default_factory=lambda: _str("WHISPER_COMPUTE_TYPE", "auto").lower())
    whisper_beam_size: int = field(default_factory=lambda: _int("WHISPER_BEAM_SIZE", 1))
    seamless_model: str = field(default_factory=lambda: _str("SEAMLESS_MODEL", "ai4bharat/indic-seamless"))
    sarvam_stt_model: str = field(default_factory=lambda: _str("SARVAM_STT_MODEL", "saaras:v3"))
    openai_stt_model: str = field(default_factory=lambda: _str("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"))

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_provider: str = field(default_factory=lambda: _str("TTS_PROVIDER", "auto").lower())
    tts_max_concurrency: int = field(default_factory=lambda: _int("TTS_MAX_CONCURRENCY", 4))
    tts_timeout: float = field(default_factory=lambda: _float("TTS_TIMEOUT", 8.0))
    tts_cache_size: int = field(default_factory=lambda: _int("TTS_CACHE_SIZE", 256))
    tts_first_chunk_min_chars: int = field(default_factory=lambda: _int("TTS_FIRST_CHUNK_MIN_CHARS", 24))
    tts_first_chunk_max_chars: int = field(default_factory=lambda: _int("TTS_FIRST_CHUNK_MAX_CHARS", 70))
    sarvam_api_key: str = field(default_factory=lambda: _str("SARVAM_API_KEY"))
    sarvam_tts_model: str = field(default_factory=lambda: _str("SARVAM_TTS_MODEL", "bulbul:v3"))
    sarvam_speaker: str = field(default_factory=lambda: _str("SARVAM_SPEAKER", "priya"))
    openai_tts_model: str = field(default_factory=lambda: _str("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
    openai_tts_voice: str = field(default_factory=lambda: _str("OPENAI_TTS_VOICE", "coral"))
    elevenlabs_api_key: str = field(default_factory=lambda: _str("ELEVENLABS_API_KEY"))
    elevenlabs_model: str = field(default_factory=lambda: _str("ELEVENLABS_MODEL", "eleven_flash_v2_5"))
    elevenlabs_voice_id: str = field(default_factory=lambda: _str("ELEVENLABS_VOICE_ID", ""))

    # ── Server ───────────────────────────────────────────────────────────────
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in _str("CORS_ORIGINS", "*").split(",") if o.strip()])
    vad_silence_ms: int = field(default_factory=lambda: _int("VAD_SILENCE_MS", 550))
    max_audio_seconds: int = field(default_factory=lambda: _int("MAX_AUDIO_SECONDS", 30))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO").upper())

    @property
    def kb_configured(self) -> bool:
        return bool(self.gcp_project_id and self.gcp_data_store_id)

    def llm_chain(self) -> list[str]:
        """Primary provider followed by de-duplicated fallbacks."""
        chain: list[str] = []
        for name in [self.llm_provider, *self.llm_fallbacks]:
            if name and name not in chain:
                chain.append(name)
        return chain


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
