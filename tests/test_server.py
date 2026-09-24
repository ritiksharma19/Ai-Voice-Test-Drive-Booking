"""WebSocket protocol test with fake engines (no models, network or keys)."""
from __future__ import annotations

import asyncio

import numpy as np
from fastapi.testclient import TestClient

import server
from stt.audio import encode_wav
from tests.test_pipeline import FakeBackend, make_orchestrator
from tts.base import Speech


class FakeSTT:
    async def transcribe(self, samples, sample_rate=16000, language=None):
        return {"text": "आज मौसम कैसा है", "language": "ur"}

    def describe(self):
        return "fake-stt"

    async def close(self):
        pass


class FakeTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, language="en"):
        self.calls.append((text, language))
        await asyncio.sleep(0.01)
        return Speech("QUJD", "mp3")

    def describe(self):
        return "fake-tts"


def _install_fakes(tokens):
    server.engines.stt = FakeSTT()
    server.engines.tts = FakeTTS()
    server.engines.llm = make_orchestrator(FakeBackend("fake", tokens=tokens))
    return server.engines.tts


def _drain(ws):
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg["type"] in ("metrics", "error"):
            return msgs


def test_audio_turn_end_to_end():
    tts = _install_fakes(("आज मौसम साफ़ है, ", "धूप खिली रहेगी। ", "शाम को हल्की ठंड होगी।"))
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(encode_wav(np.zeros(16_000, np.float32)))
        msgs = _drain(ws)
    types = [m["type"] for m in msgs]
    assert types[0] == "transcription" and types[1] == "status"
    audio = [m for m in msgs if m["type"] == "audio_chunk"]
    assert [a["text"] for a in audio] == ["आज मौसम साफ़ है, धूप खिली रहेगी।", "शाम को हल्की ठंड होगी।"]
    assert all(lang == "hi" for _, lang in tts.calls)          # Urdu STT tag → Hindi voice
    done = next(m for m in msgs if m["type"] == "done")
    assert done["source"] == "none"
    timings = msgs[-1]["timings_ms"]
    assert {"stt", "llm_ttft", "first_audio", "total"} <= timings.keys()


def test_booking_event_reaches_client():
    from tests.test_business import BOOK_CALL, ScriptedBackend, bookings
    server.engines.stt, server.engines.tts = FakeSTT(), FakeTTS()
    server.engines.llm = make_orchestrator(
        ScriptedBackend(["One moment. ", BOOK_CALL], ["You're booked."]), bookings=bookings())
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "yes, book it"})
        msgs = _drain(ws)
    booking = next(m for m in msgs if m["type"] == "booking")["booking"]
    assert booking["booking_id"].startswith("TD-") and booking["phone_last4"] == "3210"
    spoken = " ".join(m["text"] for m in msgs if m["type"] == "audio_chunk")
    assert "action" not in spoken and "You're booked." in spoken
    assert client.get("/bookings").status_code == 404            # ADMIN_TOKEN not set


def test_text_turn_and_health():
    _install_fakes(("Hello there. ", "How can I help?"))
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "hello"})
        msgs = _drain(ws)
    assert [m["text"] for m in msgs if m["type"] == "audio_chunk"] == ["Hello there.", "How can I help?"]
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["engines"]["stt"] == "fake-stt"
    assert "llm_ttft" in client.get("/metrics").json()
    assert client.get("/config").json()["vad_silence_ms"] > 0
