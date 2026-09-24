"""
core/metrics.py
Lightweight in-process latency tracking (no external dependencies).

Each conversational turn records named stage timings (ms). Recent samples are
kept in bounded ring buffers so /metrics can report p50/p95/max per stage.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

_WINDOW = 500


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, round(pct / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


class LatencyStats:
    def __init__(self, window: int = _WINDOW) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counts: dict[str, int] = defaultdict(int)

    def record(self, stage: str, ms: float) -> None:
        self._samples[stage].append(ms)
        self._counts[stage] += 1

    def snapshot(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, values in self._samples.items():
            vals = sorted(values)
            out[stage] = {
                "count": self._counts[stage],
                "p50_ms": round(_percentile(vals, 50), 1),
                "p95_ms": round(_percentile(vals, 95), 1),
                "max_ms": round(vals[-1], 1) if vals else 0.0,
            }
        return out


STATS = LatencyStats()


@dataclass
class TurnTimer:
    """Marks elapsed time from turn start; each mark is recorded once."""
    t0: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def mark(self, stage: str, ms: float | None = None) -> float:
        if stage in self.marks:
            return self.marks[stage]
        value = self.elapsed_ms() if ms is None else ms
        self.marks[stage] = value
        STATS.record(stage, value)
        return value

    def summary(self) -> dict[str, float]:
        return {k: round(v, 1) for k, v in self.marks.items()}
