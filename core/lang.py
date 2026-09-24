"""
core/lang.py
Language utilities shared by STT, LLM and TTS.

Language identification is script-first: Indic scripts map deterministically
to a language in microseconds, which is both faster and more accurate than a
statistical detector on short utterances. Lingua (optional) is consulted only
to split Devanagari between Hindi and Marathi.
"""
from __future__ import annotations

import re
from functools import lru_cache

from config.logging_config import get_logger

logger = get_logger("core.lang")

SUPPORTED = ("en", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa")

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (Devanagari script only — never Urdu/Arabic script)",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi (Gurmukhi script)",
}

# Languages Whisper confuses with a supported one → the language we serve.
LANGUAGE_ALIASES = {"ur": "hi", "ne": "hi", "sa": "hi", "as": "bn", "or": "hi"}

# (first codepoint, last codepoint, language)
_SCRIPT_RANGES = (
    (0x0900, 0x097F, "hi"),   # Devanagari (hi / mr — refined below)
    (0x0980, 0x09FF, "bn"),   # Bengali-Assamese
    (0x0A00, 0x0A7F, "pa"),   # Gurmukhi
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
    (0x0600, 0x06FF, "ur"),   # Arabic script (Urdu)
)

INDIC_SCRIPT_RE = re.compile(r"[ऀ-ൿ]")


def normalize_language(code: str | None, default: str = "en") -> str:
    code = (code or "").lower().split("-")[0]
    code = LANGUAGE_ALIASES.get(code, code)
    return code if code in SUPPORTED else default


def script_language(text: str) -> str | None:
    """Dominant-script language of `text`, or None for Latin/unknown."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if cp < 0x0600:
            continue
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.__getitem__)


@lru_cache(maxsize=1)
def _devanagari_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder  # type: ignore
        return (LanguageDetectorBuilder
                .from_languages(Language.HINDI, Language.MARATHI)
                .with_preloaded_language_models()
                .build())
    except ImportError:
        return None


def warmup() -> None:
    """Load Lingua models at startup so the first request does not pay for it."""
    if _devanagari_detector() is not None:
        logger.info("Lingua Hindi/Marathi disambiguator ready")


def detect_language(text: str, stt_hint: str | None = None) -> str:
    """
    Final language for a transcript.
      Indic script → that language (Devanagari: Hindi vs Marathi via Lingua / STT hint)
      Arabic script → Hindi (Whisper often writes Hindi speech in Urdu script)
      Latin script → STT hint if it is a supported language, else English
                     (romanised Hindi is answered in the language the STT heard)
    """
    hint = normalize_language(stt_hint, default="") or None
    lang = script_language(text)
    if lang == "ur":
        return "hi"
    if lang == "hi":
        if hint == "mr":
            return "mr"
        detector = _devanagari_detector()
        if detector is not None:
            try:
                found = detector.detect_language_of(text)
                if found is not None:
                    return found.iso_code_639_1.name.lower()
            except Exception as exc:  # pragma: no cover
                logger.debug("Lingua error: %s", exc)
        return "hi"
    if lang:
        return lang
    return hint or "en"


# ── TTS text hygiene ─────────────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FFFF\U00002702-\U000027B0☀-⛿✀-➿]+"
)
_MARKDOWN_RE = re.compile(r"[*_`#~|>]+")
_LINK_RE = re.compile(r"https?://\S+")


def clean_for_speech(text: str) -> str:
    text = _LINK_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return text.strip("()[]{} \n\t")
