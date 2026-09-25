"""
llm/orchestrator.py
LLMOrchestrator — turns a customer utterance into a low-latency text stream.

  • Provider chain (LLM_PROVIDER + LLM_FALLBACKS) across Gemini, OpenAI,
    Anthropic and Ollama. Fails over when a provider errors or does not
    produce a first token within LLM_FIRST_TOKEN_TIMEOUT. A provider that
    just failed is skipped for a short cool-down so later turns don't pay
    the same timeout again.
  • Topic guard (llm/topic_guard.py): only car / dealership messages may use
    web search; anything else with no knowledge-base match is sent to the
    model with a [TOPIC CHECK] note and politely declined.
  • Retrieval: knowledge base first, Google second (llm/retrieval.py). If a
    web lookup is still running after RETRIEVAL_FILLER_AFTER seconds, a short
    "let me check" is spoken in the customer's language so the line never
    goes silent.
  • Booking: the model calls check_availability / book_appointment with an
    <action> tag (booking/actions.py). The server validates and runs it, feeds
    the [ACTION RESULT] back, and the model speaks the outcome — at most
    _MAX_ACTION_ROUNDS actions per customer turn.
  • Bounded concurrency across all sessions (LLM_MAX_CONCURRENCY).
  • Per-session history trimmed by turns and characters; partial replies
    are kept when the customer barges in, so the model knows what was said.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import AsyncIterator

from booking import ActionFilter, BookingService, looks_like_booking
from config.logging_config import get_logger
from config.settings import Settings, get_settings
from core.metrics import TurnTimer
from core.privacy import contains_contact_details
from llm.base import LLMBackend
from llm.prompts import FILLERS, build_action_result, build_system_prompt, build_user_message
from llm.providers import build_backend
from llm.retrieval import RetrievalService, is_conversational
from llm.topic_guard import TopicGuard

logger = get_logger("llm.orchestrator")

_COOLDOWN_S = 20.0
_MAX_SESSIONS = 5000
_MAX_ACTION_ROUNDS = 2          # e.g. check_availability, then book_appointment
_BOOKING_IDLE_TURNS = 6         # booking mode ends after this many turns without booking talk


class AllProvidersFailed(RuntimeError):
    pass


class LLMOrchestrator:
    def __init__(self, settings: Settings | None = None,
                 retrieval: RetrievalService | None = None,
                 bookings: BookingService | None = None) -> None:
        self.s = settings or get_settings()
        self.retrieval = retrieval or RetrievalService(self.s)
        self.bookings = bookings or BookingService(self.s)
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
        self.booking_turns: dict[str, int] = {}   # session → turns left in booking mode
        self.guard = TopicGuard(self.s)
        self.last_topic: dict[str, tuple[str, bool]] = {}   # session → (text, was on topic)
        logger.info("LLM chain: %s", " → ".join(b.describe() for b in self.backends))

    # ── lifecycle ─────────────────────────────────────────────────────────────

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
                old, _ = self.conversations.popitem(last=False)
                self.booking_turns.pop(old, None)
                self.last_topic.pop(old, None)
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
        self.booking_turns.pop(session_id, None)
        self.last_topic.pop(session_id, None)

    def _booking_active(self, session_id: str, user_text: str) -> bool:
        if looks_like_booking(user_text):
            self.booking_turns[session_id] = _BOOKING_IDLE_TURNS
        left = self.booking_turns.get(session_id, 0)
        if left <= 0:
            return False
        self.booking_turns[session_id] = left - 1
        return True

    # ── main entry point ──────────────────────────────────────────────────────

    async def stream_reply(
        self,
        user_text: str,
        session_id: str,
        language: str = "en",
        timer: TurnTimer | None = None,
    ) -> AsyncIterator[dict]:
        """Yield {"text", "source", "provider"} chunks, plus {"booking": {...}}
        once when a booking is confirmed."""
        timer = timer or TurnTimer()
        history = list(self._history(session_id))
        self._append(session_id, "user", user_text)

        booking = self._booking_active(session_id, user_text)
        on_topic = (booking or is_conversational(user_text)
                    or contains_contact_details(user_text) or self.guard.is_on_topic(user_text))
        search_text = user_text
        prev_text, prev_on_topic = self.last_topic.get(session_id, ("", False))
        if prev_on_topic and len(user_text.split()) <= 8 and not booking:
            # Short follow-up ("and how far does it go?"): search it together
            # with the previous question. Web search still needs this message
            # itself to be on topic.
            search_text = f"{prev_text} {user_text}"
        sources = self.retrieval.plan(search_text, booking, allow_web=on_topic)
        spoken: list[str] = []          # text of the current round, for history
        retrieval = asyncio.create_task(
            self.retrieval.get_context(search_text, self.s.retrieval_wait, sources))
        try:
            if "web" in sources:
                done, _ = await asyncio.wait({retrieval}, timeout=self.s.retrieval_filler_after)
                if not done:
                    filler = FILLERS.get(language, FILLERS["en"])
                    timer.mark("filler")
                    spoken.append(filler + " ")
                    yield {"text": filler + " ", "source": "none", "provider": "filler"}
            found = await retrieval
        finally:
            if not retrieval.done():
                retrieval.cancel()
        timer.mark("retrieval_done")
        source = found["source"]
        topic_check = not on_topic and source == "none"
        if topic_check:
            logger.info("Topic guard: no car/dealership match — model asked to check scope")
        self.last_topic[session_id] = (search_text[-200:], on_topic or source != "none")

        messages = [*history, {"role": "user", "content": build_user_message(
            user_text, source, found["context"], topic_check=topic_check)}]
        system = build_system_prompt(self.s, language, self.bookings.now())

        llm_started = False
        provider = ""
        try:
            for round_no in range(_MAX_ACTION_ROUNDS + 1):
                action_filter = ActionFilter()
                async with self._sem:
                    async for provider, text in self._stream_with_failover(system, messages):
                        if not llm_started:
                            timer.mark("llm_ttft")
                            llm_started = True
                        say = action_filter.feed(text)
                        if say:
                            spoken.append(say)
                            yield {"text": say, "source": source, "provider": provider}
                say = action_filter.flush()
                if say:
                    spoken.append(say)
                    yield {"text": say, "source": source, "provider": provider}

                if not action_filter.called:
                    break
                if round_no == _MAX_ACTION_ROUNDS:
                    logger.warning("Action limit reached in one turn — ignoring %s",
                                   action_filter.raw_tag[:80])
                    break

                action = action_filter.action or {}
                result = (await self.bookings.execute(action, session_id, language) if action
                          else {"status": "error", "error": "invalid_action_format",
                                "message": "The action tag must contain valid JSON."})
                logger.info("Action %s → %s", action.get("name", "?"), result.get("status"))
                timer.mark(f"action_{round_no + 1}")
                if result.get("status") == "confirmed":
                    self.booking_turns.pop(session_id, None)
                    if not result.get("already_booked"):
                        yield {"text": "", "source": source, "provider": "booking",
                               "booking": result}

                call = "".join(spoken) + action_filter.raw_tag
                spoken.clear()
                result_msg = build_action_result(result)
                self._append(session_id, "assistant", call)
                self._append(session_id, "user", result_msg)
                messages += [{"role": "assistant", "content": call},
                             {"role": "user", "content": result_msg}]
        finally:
            # Also runs on barge-in (cancellation): keep what was actually said.
            if spoken:
                self._append(session_id, "assistant", "".join(spoken))

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
