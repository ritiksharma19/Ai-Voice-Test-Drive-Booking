"""
core/chunker.py
Splits a streaming LLM reply into speakable segments for TTS.

The first segment is cut as early as it sounds natural (clause boundary after
TTS_FIRST_CHUNK_MIN_CHARS, or a word boundary at TTS_FIRST_CHUNK_MAX_CHARS),
because time-to-first-audio is what the user perceives. Later segments are whole
sentences, which gives smoother prosody. Tiny fragments ("Yes.", "हाँ।") are
merged into the next segment instead of being dropped.
"""
from __future__ import annotations

import re

from core.lang import clean_for_speech

# Sentence end: . ! ? followed by whitespace (so "3.5" and a dot at the very end
# of the buffer are never split early), or Indic danda / newline.
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*\s+|[।॥\n]+\s*")
_CLAUSE_END = re.compile(r"[,;:–—]\s+")


class SpeechChunker:
    def __init__(self, first_min: int = 24, first_max: int = 70,
                 max_len: int = 260, min_len: int = 6) -> None:
        self.first_min, self.first_max = first_min, first_max
        self.max_len, self.min_len = max_len, min_len
        self.buf = ""
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            self._emit(cut, out)
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        if self.buf.strip():
            self._emit(len(self.buf), out)
        self.buf = ""
        return out

    # ── internals ─────────────────────────────────────────────────────────────

    def _find_cut(self) -> int | None:
        buf = self.buf
        for m in _SENTENCE_END.finditer(buf):
            if len(buf[:m.end()].strip()) >= self.min_len:
                return m.end()
        if self.emitted == 0:
            for m in _CLAUSE_END.finditer(buf):
                if m.start() >= self.first_min:
                    return m.end()
            if len(buf) >= self.first_max:
                return self._word_cut(self.first_max)
        elif len(buf) >= self.max_len:
            clauses = [m.end() for m in _CLAUSE_END.finditer(buf, 0, self.max_len)]
            return clauses[-1] if clauses else self._word_cut(self.max_len)
        return None

    def _word_cut(self, limit: int) -> int:
        space = self.buf.rfind(" ", 0, limit)
        return space + 1 if space > self.min_len else limit

    def _emit(self, cut: int, out: list[str]) -> None:
        segment, self.buf = self.buf[:cut], self.buf[cut:]
        speech = clean_for_speech(segment)
        if speech and any(c.isalnum() for c in speech):
            out.append(speech)
            self.emitted += 1
