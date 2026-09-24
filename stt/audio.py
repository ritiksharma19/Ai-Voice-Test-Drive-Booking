"""
stt/audio.py
Allocation-light PCM helpers shared by every STT backend.
No ffmpeg / pydub: the browser sends 16-bit PCM WAV, parsed directly with numpy.
"""
from __future__ import annotations

import struct

import numpy as np

SAMPLE_RATE = 16_000


def parse_wav(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV bytes → (float32 mono samples in [-1, 1], sample_rate)."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("expected a 16-bit PCM WAV frame")
    channels = struct.unpack_from("<H", data, 22)[0]
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    bits = struct.unpack_from("<H", data, 34)[0]
    if bits != 16:
        raise ValueError(f"unsupported WAV bit depth: {bits}")
    idx = data.index(b"data", 12)
    size = struct.unpack_from("<I", data, idx + 4)[0]
    pcm = np.frombuffer(data, dtype="<i2", count=min(size, len(data) - idx - 8) // 2,
                        offset=idx + 8)
    samples = pcm.astype(np.float32) * (1.0 / 32768.0)
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Linear-interpolation resample to 16 kHz (browser already sends 16 kHz)."""
    samples = np.asarray(samples, dtype=np.float32)
    if sample_rate == SAMPLE_RATE or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * SAMPLE_RATE / sample_rate))
    x_out = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
    return np.interp(x_out, np.arange(len(samples)), samples).astype(np.float32)


def encode_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 samples → 16-bit PCM WAV bytes (for cloud STT uploads)."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(pcm),
    )
    return header + pcm


def trim_silence(samples: np.ndarray, threshold: float = 0.35, pad_ms: int = 200) -> np.ndarray:
    """
    Crop leading/trailing silence with Silero VAD (ONNX, bundled with
    faster-whisper — no torch needed). Returns an empty array for pure silence.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps  # type: ignore

    spans = get_speech_timestamps(samples, VadOptions(threshold=threshold, min_silence_duration_ms=300))
    if not spans:
        return samples[:0]
    pad = SAMPLE_RATE * pad_ms // 1000
    start = max(0, spans[0]["start"] - pad)
    end = min(len(samples), spans[-1]["end"] + pad)
    return np.ascontiguousarray(samples[start:end])


_HALLUCINATIONS = frozenset({
    "thank you", "thank you.", "thanks for watching", "thank you for watching",
    "subtitles by", "you", ".", "ammen", "ammen.", "bye.",
})


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    return t in _HALLUCINATIONS or len(t.replace(".", "").strip()) <= 1
