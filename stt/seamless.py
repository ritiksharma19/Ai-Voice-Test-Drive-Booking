"""
stt/seamless.py
AI4Bharat Indic-Seamless (SeamlessM4T-v2) speech-to-text on CUDA (fp16).

Optional backend (STT_PROVIDER=seamless): needs `pip install -r requirements-seamless.txt`
(PyTorch CUDA + transformers). The model needs a target language up front:
set STT_LANGUAGE, otherwise Hindi is assumed.
"""
from __future__ import annotations

import time

import numpy as np

from config.logging_config import get_logger
from config.settings import Settings
from stt.audio import is_hallucination, trim_silence
from stt.base import LocalSTT

logger = get_logger("stt.seamless")

_LANG_MAP = {
    "en": "eng", "hi": "hin", "ta": "tam", "te": "tel", "kn": "kan",
    "ml": "mal", "bn": "ben", "gu": "guj", "mr": "mar", "pa": "pan",
}


class SeamlessSTT(LocalSTT):
    name = "seamless"

    def __init__(self, s: Settings) -> None:
        super().__init__()
        import torch  # type: ignore
        from transformers import (  # type: ignore
            SeamlessM4TFeatureExtractor,
            SeamlessM4TTokenizer,
            SeamlessM4Tv2ForSpeechToText,
        )

        self._torch = torch
        self.model_id = s.seamless_model
        self.default_language = s.stt_language or "hi"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        t0 = time.perf_counter()
        logger.info("Loading %s on %s…", self.model_id, self.device)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            self.model_id, torch_dtype=dtype).to(self.device).eval()
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(self.model_id)
        self.tokenizer = SeamlessM4TTokenizer.from_pretrained(self.model_id)
        self._dtype = dtype
        self._transcribe_sync(np.zeros(16_000, dtype=np.float32) + 1e-3, self.default_language)
        logger.info("✅ Indic-Seamless ready in %.1fs", time.perf_counter() - t0)

    def describe(self) -> str:
        return f"seamless:{self.model_id}@{self.device}"

    def _transcribe_sync(self, samples: np.ndarray, language: str | None) -> dict:
        t0 = time.perf_counter()
        lang = language or self.default_language
        audio = trim_silence(samples)
        if audio.size < 1600:
            return {"text": "", "language": lang}
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self._dtype) if v.is_floating_point() else v.to(self.device)
                  for k, v in inputs.items()}
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, tgt_lang=_LANG_MAP.get(lang, "hin"))
        text = self.tokenizer.decode(out[0].cpu().numpy().squeeze(),
                                     clean_up_tokenization_spaces=True,
                                     skip_special_tokens=True).strip()
        if is_hallucination(text):
            text = ""
        logger.info("STT %.0f ms | %s | %r", (time.perf_counter() - t0) * 1000, lang, text[:80])
        return {"text": text, "language": lang}
