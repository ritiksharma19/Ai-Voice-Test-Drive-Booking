"""
core/vad.py
Server-side streaming voice activity detection (VAD) for continuous audio.

The browser client runs its own VAD and sends whole utterances, but a phone
call arrives as an endless PCM stream, so the server has to find the turns:

  speech_start — the caller has talked for VAD_MIN_SPEECH_MS (used for barge-in)
  speech_end   — followed by VAD_SILENCE_MS of silence; carries the utterance

Scorers (VAD_PROVIDER):
  silero — Silero VAD v6 (ONNX, bundled with faster-whisper; no PyTorch), run
           one 32 ms window at a time with its LSTM state carried over.
           ~0.1 ms per window on one CPU core.                       [default]
  energy — frame RMS against an adaptive noise floor. No model; used when
           Silero cannot load.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from config.logging_config import get_logger
from config.settings import Settings
from stt.audio import SAMPLE_RATE

logger = get_logger("core.vad")

FRAME = 512                               # samples per decision at 16 kHz
FRAME_MS = FRAME * 1000 // SAMPLE_RATE    # 32 ms
_HYSTERESIS = 0.15                        # speech continues until prob < threshold - this


# ── scorers: one 32 ms float32 frame → speech probability ────────────────────

@lru_cache(maxsize=1)
def _silero_session():
    from faster_whisper.vad import get_vad_model  # type: ignore
    return get_vad_model().session


class SileroScorer:
    _CONTEXT = 64

    def __init__(self) -> None:
        self._session = _silero_session()
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(self._CONTEXT, np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame])[None, :]
        prob, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-self._CONTEXT:]
        return float(prob[0])


class EnergyScorer:
    """RMS above an adaptive noise floor, mapped to 0..1 (12 dB above floor ≈ 0.5)."""

    def __init__(self) -> None:
        self._floor = 0.002

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame * frame))) + 1e-9
        if rms < self._floor:
            self._floor = 0.8 * self._floor + 0.2 * rms       # falls fast
        else:
            self._floor *= 1.003                              # rises slowly
        self._floor = min(max(self._floor, 3e-4), 0.05)
        if rms < 0.004:                                       # ≈ -48 dBFS: silence on any line
            return 0.0
        snr_db = 20 * np.log10(rms / self._floor)
        return float(np.clip((snr_db - 6) / 12, 0.0, 1.0))


def make_scorer(provider: str):
    if provider == "silero":
        try:
            return SileroScorer()
        except Exception as exc:
            logger.warning("Silero VAD unavailable (%r) — using the energy VAD", exc)
    return EnergyScorer()


def warmup(provider: str) -> None:
    make_scorer(provider)(np.zeros(FRAME, np.float32))


# ── stream resampling (8 / 24 kHz phone audio → 16 kHz) ──────────────────────

class _StreamResampler:
    """Linear interpolation that stays continuous across chunk boundaries."""

    def __init__(self, src: int, dst: int = SAMPLE_RATE) -> None:
        self._step = src / dst
        self._pos = 0.0
        self._tail = np.zeros(0, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._step == 1.0 or len(x) == 0:
            return x
        buf = np.concatenate([self._tail, x])
        last = len(buf) - 1
        n = int(np.floor((last - self._pos) / self._step)) + 1 if last >= self._pos else 0
        idx = self._pos + np.arange(n) * self._step
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        self._pos = (idx[-1] + self._step if n else self._pos) - last
        self._tail = buf[-1:]
        return out


# ── endpointing ──────────────────────────────────────────────────────────────

@dataclass
class VADEvent:
    kind: str                           # "speech_start" | "speech_end"
    audio: np.ndarray | None = None     # 16 kHz float32 utterance on speech_end


class StreamingVAD:
    """Feed PCM chunks of any size; get speech_start / speech_end events back."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, *, provider: str = "silero",
                 threshold: float = 0.5, min_speech_ms: int = 250, silence_ms: int = 550,
                 max_utterance_s: int = 30, pre_roll_ms: int = 300) -> None:
        self._resample = _StreamResampler(sample_rate)
        self._score = make_scorer(provider)
        self.provider = "silero" if isinstance(self._score, SileroScorer) else "energy"
        self.threshold = threshold
        self._min_speech = max(1, min_speech_ms // FRAME_MS)
        self._min_silence = max(1, silence_ms // FRAME_MS)
        self._max_frames = max_utterance_s * 1000 // FRAME_MS
        self._pre_roll = pre_roll_ms // FRAME_MS
        self._pending = np.zeros(0, np.float32)
        self._frames: deque[np.ndarray] = deque()
        self._speech_run = 0
        self._silence_run = 0
        self.in_speech = False

    @classmethod
    def from_settings(cls, s: Settings, sample_rate: int) -> "StreamingVAD":
        return cls(sample_rate, provider=s.vad_provider, threshold=s.vad_threshold,
                   min_speech_ms=s.vad_min_speech_ms, silence_ms=s.vad_silence_ms,
                   max_utterance_s=s.max_audio_seconds)

    def reset(self) -> None:
        self._frames.clear()
        self._speech_run = self._silence_run = 0
        self.in_speech = False

    def feed(self, samples: np.ndarray) -> list[VADEvent]:
        """`samples`: float32 mono in [-1, 1] at the constructor's sample rate."""
        self._pending = np.concatenate([self._pending, self._resample(samples)])
        n = len(self._pending) // FRAME
        frames, self._pending = self._pending[:n * FRAME], self._pending[n * FRAME:]
        events: list[VADEvent] = []
        for frame in frames.reshape(-1, FRAME):
            event = self._step(frame, self._score(frame))
            if event:
                events.append(event)
        return events

    def _step(self, frame: np.ndarray, prob: float) -> VADEvent | None:
        self._frames.append(frame)
        if not self.in_speech:
            self._speech_run = self._speech_run + 1 if prob >= self.threshold else 0
            while len(self._frames) > self._pre_roll + self._speech_run:
                self._frames.popleft()
            if self._speech_run >= self._min_speech:
                self.in_speech, self._silence_run = True, 0
                return VADEvent("speech_start")
            return None

        self._silence_run = self._silence_run + 1 if prob < self.threshold - _HYSTERESIS else 0
        if self._silence_run >= self._min_silence or len(self._frames) >= self._max_frames:
            keep = len(self._frames) - max(0, self._silence_run - self._pre_roll)
            audio = np.concatenate(list(self._frames)[:keep])
            self.reset()
            return VADEvent("speech_end", audio)
        return None
