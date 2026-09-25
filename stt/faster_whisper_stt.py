"""
stt/faster_whisper_stt.py
Local Whisper on NVIDIA CUDA (float16) via faster-whisper / CTranslate2,
with an int8 CPU fallback.

Latency choices:
  • Silero VAD (ONNX, bundled) crops silence before the encoder runs.
  • Language ID is restricted to STT_LANGUAGES (e.g. Urdu → re-decoded as Hindi,
    other unsupported languages → the most likely supported one).
  • Greedy decoding (beam 1), no timestamps, no conditioning on previous text,
    bounded max_new_tokens.
"""
from __future__ import annotations

import time

import numpy as np

from config.device import get_whisper_device, register_cuda_dlls
from config.logging_config import get_logger
from config.settings import Settings
from core.lang import normalize_language
from stt.audio import is_hallucination, trim_silence
from stt.base import LocalSTT

logger = get_logger("stt.faster_whisper")


class FasterWhisperSTT(LocalSTT):
    name = "faster-whisper"

    def __init__(self, s: Settings) -> None:
        super().__init__(s.whisper_workers)
        register_cuda_dlls()
        from faster_whisper import WhisperModel  # type: ignore

        self.device, compute_type = get_whisper_device(s.whisper_device, s.whisper_compute_type)
        # large-v3-turbo: near large-v3 accuracy at ~8x the speed — the best GPU default.
        self.model_name = s.whisper_model or ("large-v3-turbo" if self.device == "cuda" else "small")
        if self.model_name.startswith("mlx-community/"):
            raise ValueError(
                f"WHISPER_MODEL={self.model_name} is an Apple MLX model. Use a faster-whisper "
                "model such as large-v3-turbo, medium or small.")
        self.forced_language = s.stt_language or None
        self.allowed = set(s.stt_languages)
        self.beam_size = s.whisper_beam_size

        t0 = time.perf_counter()
        logger.info("Loading faster-whisper %s on %s (%s)…", self.model_name, self.device, compute_type)
        # num_workers lets CTranslate2 run that many transcribe() calls in parallel.
        self.model = WhisperModel(self.model_name, device=self.device, compute_type=compute_type,
                                  num_workers=s.whisper_workers)
        self.multilingual = self.model.model.is_multilingual

        # Warm-up: first CUDA inference pays kernel/JIT setup (~0.5–2 s).
        noise = (np.random.default_rng(0).standard_normal(16_000) * 0.01).astype(np.float32)
        try:
            segments, _ = self.model.transcribe(noise, language="en", beam_size=1)
            list(segments)
        except Exception:
            pass
        logger.info("✅ faster-whisper ready in %.1fs (device=%s)", time.perf_counter() - t0, self.device)

    def describe(self) -> str:
        return f"faster-whisper:{self.model_name}@{self.device}"

    def _decode(self, audio: np.ndarray, language: str | None):
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            temperature=[0.0, 0.4],
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,          # already trimmed
            max_new_tokens=220,
        )
        return "".join(seg.text for seg in segments).strip(), info

    def _best_allowed(self, probs) -> str:
        best: dict[str, float] = {}
        for code, p in probs or []:
            lang = normalize_language(code, default="")
            if lang and (not self.allowed or lang in self.allowed):
                best[lang] = best.get(lang, 0.0) + p
        return max(best, key=best.__getitem__) if best else "en"

    def _transcribe_sync(self, samples: np.ndarray, language: str | None) -> dict:
        t0 = time.perf_counter()
        audio = trim_silence(samples)
        hint = language or self.forced_language
        if audio.size < 1600:   # < 0.1 s of speech
            return {"text": "", "language": hint or "en"}

        # Auto-detect inside transcribe (single detection pass). Only when Whisper
        # picks a language outside STT_LANGUAGES (typically Urdu for Hindi speech)
        # is the audio decoded again in the most likely allowed language.
        text, info = self._decode(audio, hint if self.multilingual else "en")
        lang = info.language
        if not hint and self.allowed and lang not in self.allowed:
            lang = self._best_allowed(info.all_language_probs)
            text, _ = self._decode(audio, lang)
        if is_hallucination(text):
            text = ""
        logger.info("STT %.0f ms | %s | %r", (time.perf_counter() - t0) * 1000, lang, text[:80])
        return {"text": text, "language": lang}
