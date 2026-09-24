"""
Offline tests — no GPU, network or API keys required.
Run:  python -m pytest -q
"""
from __future__ import annotations

import asyncio
import struct
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from booking import BookingService
from config.settings import Settings
from core.chunker import SpeechChunker
from core.lang import detect_language, script_language
from llm.base import LLMBackend
from llm.orchestrator import AllProvidersFailed, LLMOrchestrator
from llm.retrieval import RetrievalService
from stt.audio import encode_wav, parse_wav, to_16k


def base_settings(**overrides) -> Settings:
    s = Settings()
    defaults = dict(llm_provider="fake", llm_fallbacks=[], web_search_enabled=False,
                    gcp_project_id="", gcp_data_store_id="", retrieval_mode="auto",
                    kb_provider="off", llm_first_token_timeout=0.3,
                    booking_db_path=str(Path(tempfile.mkdtemp()) / "bookings.db"))
    defaults.update(overrides)
    return replace(s, **defaults)


# ── chunker ──────────────────────────────────────────────────────────────────

def feed_all(chunker: SpeechChunker, text: str, step: int = 3) -> list[str]:
    out = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i:i + step])
    return out + chunker.flush()


def test_first_chunk_cut_at_clause():
    text = "Solar panels convert sunlight into electricity, using photovoltaic cells. They last decades."
    parts = feed_all(SpeechChunker(first_min=24, first_max=70), text)
    assert parts[0] == "Solar panels convert sunlight into electricity,"
    assert parts[-1] == "They last decades."


def test_decimals_not_split_and_short_fragments_merged():
    parts = feed_all(SpeechChunker(), "Yes. The price is 10.5 lakh rupees today. Thanks!")
    joined = " ".join(parts)
    assert "10.5" in joined
    assert parts[0].startswith("Yes. The price")


def test_danda_and_markdown_cleanup():
    parts = feed_all(SpeechChunker(), "**नमस्ते**, आप कैसे हैं। मैं ठीक हूँ।")
    assert parts == ["नमस्ते, आप कैसे हैं।", "मैं ठीक हूँ।"]


def test_long_first_sentence_is_cut_at_word_boundary():
    text = "a" * 5 + " " + "word " * 30
    first = SpeechChunker(first_min=24, first_max=70).feed(text)[0]
    assert len(first) <= 70 and not first.endswith("wor")


# ── language ID ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,hint,expected", [
    ("आज सोने का भाव क्या है", "ur", "hi"),
    ("இன்று வானிலை எப்படி", None, "ta"),
    ("ಇಂದು ಹವಾಮಾನ", "hi", "kn"),
    ("what is the weather", "hi", "hi"),       # romanised / English with hint
    ("what is the weather", None, "en"),
    ("آج موسم کیسا ہے", None, "hi"),           # Urdu script → answer in Hindi
])
def test_detect_language(text, hint, expected):
    assert detect_language(text, hint) == expected


def test_script_language_latin_is_none():
    assert script_language("hello world") is None


# ── audio ────────────────────────────────────────────────────────────────────

def test_wav_roundtrip_and_resample():
    tone = (0.5 * np.sin(np.linspace(0, 100, 16_000))).astype(np.float32)
    samples, sr = parse_wav(encode_wav(tone))
    assert sr == 16_000 and np.allclose(samples, tone, atol=1e-3)
    assert len(to_16k(np.zeros(48_000, np.float32), 48_000)) == 16_000


def test_parse_wav_rejects_garbage():
    with pytest.raises(ValueError):
        parse_wav(b"not a wav" * 10)
    bad = bytearray(encode_wav(np.zeros(10, np.float32)))
    struct.pack_into("<H", bad, 34, 8)
    with pytest.raises(ValueError):
        parse_wav(bytes(bad))


# ── retrieval routing ────────────────────────────────────────────────────────

def test_retrieval_plan():
    no_kb = RetrievalService(base_settings(web_search_enabled=True))
    assert no_kb.plan("hi there") == ()
    assert no_kb.plan("What is photosynthesis?") == ()          # model knowledge, no wait
    assert no_kb.plan("gold price today") == ("web",)
    assert no_kb.plan("आज सोने का भाव") == ("web",)
    with_kb = RetrievalService(base_settings(web_search_enabled=True, gcp_project_id="p",
                                             gcp_data_store_id="d", kb_provider="auto"))
    assert with_kb.plan("What is our refund policy?") == ("kb", "web")
    assert with_kb.plan("हमारी रिफंड नीति क्या है?") == ()       # KB is English-only by default
    off = RetrievalService(base_settings(retrieval_mode="off", web_search_enabled=True))
    assert off.plan("gold price today") == ()


# ── orchestrator failover ────────────────────────────────────────────────────

class FakeBackend(LLMBackend):
    def __init__(self, name, tokens=("Hello", " there."), delay=0.0, fail=False):
        self.name, self.model = name, "fake"
        self.tokens, self.delay, self.fail = tokens, delay, fail
        self.calls = 0

    async def stream(self, system, messages):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        for t in self.tokens:
            yield t


def make_orchestrator(*backends, settings: Settings | None = None,
                      retrieval: RetrievalService | None = None,
                      bookings: BookingService | None = None) -> LLMOrchestrator:
    orch = LLMOrchestrator.__new__(LLMOrchestrator)
    orch.s = settings or base_settings()
    orch.retrieval = retrieval or RetrievalService(orch.s)
    orch.bookings = bookings or BookingService(orch.s)
    orch.booking_turns = {}
    orch.backends = list(backends)
    orch._cooldown_until = {}
    orch._sem = asyncio.Semaphore(4)
    from collections import OrderedDict
    orch.conversations = OrderedDict()
    return orch


async def collect(orch, text="hello", session="s1"):
    return [c async for c in orch.stream_reply(text, session)]


def test_failover_on_error_and_timeout():
    async def run():
        broken = FakeBackend("a", fail=True)
        slow = FakeBackend("b", delay=1.0)
        good = FakeBackend("c", tokens=("Hi", "!"))
        orch = make_orchestrator(broken, slow, good)
        chunks = await collect(orch)
        assert [c["provider"] for c in chunks] == ["c", "c"]
        # failed providers are cooled down → next turn goes straight to "c"
        await collect(orch, "again")
        assert broken.calls == 1 and slow.calls == 1 and good.calls == 2
        hist = orch.conversations["s1"]
        assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]
    asyncio.run(run())


def test_all_providers_failed():
    async def run():
        orch = make_orchestrator(FakeBackend("a", fail=True))
        with pytest.raises(AllProvidersFailed):
            await collect(orch)
    asyncio.run(run())


def test_partial_reply_kept_on_cancel():
    async def run():
        orch = make_orchestrator(FakeBackend("a", tokens=("One.", " Two.", " Three."), delay=0))
        agen = orch.stream_reply("count", "s2")
        await agen.__anext__()
        await agen.aclose()   # simulates barge-in
        assert orch.conversations["s2"][-1] == {"role": "assistant", "content": "One."}
    asyncio.run(run())
