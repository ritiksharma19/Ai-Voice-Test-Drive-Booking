"""
booking/actions.py
Provider-neutral "tool calling" for a streaming voice reply.

The system prompt tells the model to call a booking action by writing

    <action>{"name": "book_appointment", ...}</action>

This works identically on Gemini, OpenAI, Claude and local Ollama models, keeps
token streaming, and keeps LLM failover working (native tool-call formats differ
per provider). ActionFilter sits between the LLM stream and TTS: text before the
tag is spoken as it streams, the tag itself is never spoken, and a partial "<ac"
at the end of a chunk is held back until it is clear whether it opens a tag.
"""
from __future__ import annotations

import json
import re

OPEN, CLOSE = "<action>", "</action>"
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ActionFilter:
    def __init__(self) -> None:
        self._buf = ""
        self._in_action = False
        self._done = False
        self._payload: list[str] = []
        self.raw_tag = ""            # the full <action>…</action> text, for history

    def feed(self, text: str) -> str:
        """Speakable part of `text` (may be '' while a possible tag is buffered)."""
        if self._done:
            return ""                # the model is told to stop after the tag
        self._buf += text
        if self._in_action:
            return self._consume_action()
        start = self._buf.find(OPEN)
        if start >= 0:
            speak, self._buf = self._buf[:start], self._buf[start + len(OPEN):]
            self._in_action = True
            self._consume_action()
            return speak
        hold = _partial_suffix(self._buf, OPEN)
        speak = self._buf[:len(self._buf) - hold]
        self._buf = self._buf[len(speak):]
        return speak

    def flush(self) -> str:
        """End of stream: release held text, or finish an unterminated tag."""
        if self._in_action and not self._done:
            self._payload.append(self._buf)
            self._buf = ""
            self._finish()
            return ""
        speak, self._buf = ("" if self._done else self._buf), ""
        return speak

    def _consume_action(self) -> str:
        end = self._buf.find(CLOSE)
        if end < 0:
            keep = _partial_suffix(self._buf, CLOSE)
            self._payload.append(self._buf[:len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep:]
            return ""
        self._payload.append(self._buf[:end])
        self._buf = ""
        self._finish()
        return ""

    def _finish(self) -> None:
        self._done = True
        self.raw_tag = OPEN + "".join(self._payload) + CLOSE

    @property
    def called(self) -> bool:
        return self._done

    @property
    def action(self) -> dict | None:
        """Parsed action, {} if the tag held invalid JSON, None if there was no tag."""
        if not self._done:
            return None
        body = "".join(self._payload).strip().strip("`")
        match = _JSON_OBJECT_RE.search(body)
        try:
            data = json.loads(match.group() if match else body)
        except (json.JSONDecodeError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0
