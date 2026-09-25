"""
telephony/audio.py
PCM helpers for phone lines: raw 16-bit little-endian mono ("slin").

TTS engines return MP3; phone networks want raw PCM at the call's sample
rate. PyAV (installed with faster-whisper) decodes and resamples in one pass,
so no ffmpeg binary is needed.
"""
from __future__ import annotations

import io

import numpy as np

BYTES_PER_SAMPLE = 2


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) * (1.0 / 32768.0)


def decode_to_pcm16(audio: bytes, sample_rate: int) -> bytes:
    """Any audio PyAV can read (MP3, WAV, …) → raw 16-bit mono PCM at `sample_rate`."""
    import av  # type: ignore

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    out = bytearray()
    with av.open(io.BytesIO(audio), mode="r", metadata_errors="ignore") as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                out += resampled.to_ndarray().tobytes()
    for resampled in resampler.resample(None):   # flush
        out += resampled.to_ndarray().tobytes()
    return bytes(out)


def split_frames(pcm: bytes, chunk_bytes: int, align: int = 320, min_bytes: int = 0) -> list[bytes]:
    """Cut PCM into `chunk_bytes` pieces. The last one is zero-padded to at
    least `min_bytes` and to a multiple of `align`."""
    chunks = [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
    if chunks:
        size = max(len(chunks[-1]), min_bytes)
        size += -size % align
        chunks[-1] += b"\x00" * (size - len(chunks[-1]))
    return chunks
