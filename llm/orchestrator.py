"""
llm/orchestrator.py
LLMOrchestrator — turns a user utterance into a low-latency text stream.

  • Provider chain (LLM_PROVIDER + LLM_FALLBACKS) across Gemini, OpenAI,
    Anthropic and Ollama. Fails over when a provider errors or does not
    produce a first token within LLM_FIRST_TOKEN_TIMEOUT. A provider that
    just failed is skipped for a short cool-down so later turns don't pay
    the same timeout again.
  • Retrieval is started only when the query needs it and is awaited for at
    most LLM_RETRIEVAL_WAIT seconds.
  • Bounded concurrency across all sessions (LLM_MAX_CONCURRENCY).
  • Per-session history trimmed by turns and characters; partial replies
    are kept when the user barges in, so the model knows what was said.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import AsyncIterator

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from core.lang import LANGUAGE_NAMES
from core.metrics import TurnTimer
from llm.base import LLMBackend
from llm.providers import build_backend
from llm.retrieval import RetrievalService

logger = get_logger("llm.orchestrator")

_SYSTEM_PROMPT = """\
You are VoiceAgent, a professional real-time voice assistant.

VOICE OUTPUT RULES — your text is spoken aloud by a TTS engine:
- No markdown, asterisks, bullet points, lists, URLs, emojis or special formatting.
- Natural spoken sentences only. Keep answers to one to four short sentences unless the user asks for more detail.
- Start with the answer itself; no filler such as "Great question".
- Write numbers, units and symbols the way they are spoken ("percent", not "%").

KNOWLEDGE:
- When the user message includes retrieved context, rely on it for facts and figures.
- Otherwise answer from your own knowledge; if unsure, say so briefly and still help.
- Never mention retrieval systems, knowledge bases, search engines or tools.

TONE: warm, calm and professional; mirror the user's mood (empathetic when they are upset, upbeat when they are happy).

LANGUAGE: Reply ONLY in {lang_name}."""

_CONTEXT_TEMPLATE = """\
[Context — {label}]
{context}

[User]: {query}"""

_COOLDOWN_S = 20.0
_MAX_SESSIONS = 5000


class AllProvidersFailed(RuntimeError):
    pass


class LLMOrchestrator:
    def __init__(self, settings: Settings | None = None,
                 retrieval: RetrievalService | None = None) -> None:
        self.s = settings or get_settings()
        self.retrieval = retrieval or RetrievalService(self.s)
        self.backends: list[LLMBackend] = []
        for name in self.s.llm_chain():
            backend = build_backend(name, self.s)
            if backend.available:
                self.backends.append(backend)
            else:
                logger.warning("LLM provider '%s' skipped — API key not configured", name)
        if not self.backends:
            raise RuntimeError(
                "No usable LLM provider. Set LLM_PROVIDER and its API key in .env "
                "(or LLM_PROVIDER=ollama for local inference).")
        self._cooldown_until: dict[str, float] = {}
        self._sem = asyncio.Semaphore(self.s.llm_max_concurrency)
        self.conversations: OrderedDict[str, list[dict]] = OrderedDict()
        logger.info("LLM chain: %s", " → ".join(b.describe() for b in self.backends))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @property
    def primary(self) -> LLMBackend:
        return self.backends[0]

    async def warmup(self) -> None:
        """Warm every backend concurrently (TLS handshakes / model load)."""
        async def _one(b: LLMBackend) -> None:
            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(b.warmup(), timeout=60)
                logger.info("✅ %s warm (%.0f ms)", b.describe(), (time.perf_counter() - t0) * 1000)
            except Exception as exc:
                logger.warning("Warmup failed for %s: %s", b.describe(), exc)
        await asyncio.gather(*(_one(b) for b in self.backends), self.retrieval.warmup())

    async def close(self) -> None:
        await asyncio.gather(*(b.close() for b in self.backends), return_exceptions=True)

    # ── history ───────────────────────────────────────────────────────────────

    def _history(self, session_id: str) -> list[dict]:
        hist = self.conversations.get(session_id)
        if hist is None:
            hist = self.conversations[session_id] = []
            while len(self.conversations) > _MAX_SESSIONS:
                self.conversations.popitem(last=False)
        else:
            self.conversations.move_to_end(session_id)
        return hist

    def _append(self, session_id: str, role: str, text: str) -> None:
        hist = self._history(session_id)
        hist.append({"role": role, "content": text})
        del hist[:-self.s.history_turns * 2]
        total = sum(len(m["content"]) for m in hist)
        while total > self.s.history_chars and len(hist) > 2:
            total -= len(hist.pop(0)["content"])
        while hist and hist[0]["role"] != "user":   # providers expect a user turn first
            hist.pop(0)

    def release_session(self, session_id: str) -> None:
        self.conversations.pop(session_id, None)

    # ── main entry point ──────────────────────────────────────────────────────

    async def stream_reply(
        self,
        user_text: str,
        session_id: str,
        language: str = "en",
        timer: TurnTimer | None = None,
    ) -> AsyncIterator[dict]:
        """Yield {"text": str, "source": "kb"|"web"|"none", "provider": str} chunks."""
        timer = timer or TurnTimer()
        history = list(self._history(session_id))
        self._append(session_id, "user", user_text)

        retrieval = await self.retrieval.get_context(user_text, wait=self.s.retrieval_wait)
        timer.mark("retrieval_done")
        source, context = retrieval["source"], retrieval["context"]

        final_user = user_text
        if context:
            label = "company knowledge base" if source == "kb" else "live web results"
            final_user = _CONTEXT_TEMPLATE.format(label=label, context=context, query=user_text)
        messages = [*history, {"role": "user", "content": final_user}]
        system = _SYSTEM_PROMPT.format(lang_name=LANGUAGE_NAMES.get(language, language))

        reply: list[str] = []
        try:
            async with self._sem:
                async for provider, text in self._stream_with_failover(system, messages):
                    if not reply:
                        timer.mark("llm_ttft")
                    reply.append(text)
                    yield {"text": text, "source": source, "provider": provider}
        finally:
            # Also runs on barge-in (cancellation): keep what was actually said.
            if reply:
                self._append(session_id, "assistant", "".join(reply))

    async def _stream_with_failover(self, system: str, messages: list[dict]
                                    ) -> AsyncIterator[tuple[str, str]]:
        now = time.monotonic()
        ready = [b for b in self.backends if self._cooldown_until.get(b.name, 0) <= now]
        cooling = [b for b in self.backends if b not in ready]
        errors: list[str] = []

        for backend in ready + cooling:   # cooling-down providers only as a last resort
            agen = backend.stream(system, messages).__aiter__()
            try:
                # asyncio.timeout keeps the generator in the current task, which
                # SDK streams holding anyio cancel scopes require.
                async with asyncio.timeout(self.s.llm_first_token_timeout):
                    first = await agen.__anext__()
            except StopAsyncIteration:
                errors.append(f"{backend.name}: empty response")
                continue
            except asyncio.CancelledError:
                await agen.aclose()
                raise
            except Exception as exc:
                reason = "first-token timeout" if isinstance(exc, asyncio.TimeoutError) else repr(exc)
                logger.warning("LLM %s failed before first token (%s) — failing over",
                               backend.describe(), reason)
                errors.append(f"{backend.name}: {reason}")
                self._cooldown_until[backend.name] = time.monotonic() + _COOLDOWN_S
                await agen.aclose()
                continue

            self._cooldown_until.pop(backend.name, None)
            yield backend.name, first
            try:
                async for text in agen:
                    yield backend.name, text
            except Exception as exc:
                # Audio for the partial answer is already playing; restarting on
                # another provider would repeat it, so end the turn here.
                logger.error("LLM %s failed mid-stream: %r", backend.describe(), exc)
            finally:
                await agen.aclose()
            return

        raise AllProvidersFailed("; ".join(errors) or "no providers")
