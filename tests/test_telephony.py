"""Server-side VAD and the Exotel stream, with fake engines (no models, network or keys)."""
from __future__ import annotations

import base64
from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient

import server
from core.vad import FRAME, StreamingVAD, _StreamResampler
from stt.audio import encode_wav
from telephony.audio import decode_to_pcm16, split_frames
from tests.test_server import FakeSTT
from tests.test_pipeline import FakeBackend, make_orchestrator
from tts.base import Speech

RNG = np.random.default_rng(0)


def noise(seconds: float, rate: int, amp: float = 0.3) -> np.ndarray:
    return (RNG.uniform(-amp, amp, int(seconds * rate))).astype(np.float32)


def silence(seconds: float, rate: int) -> np.ndarray:
    return np.zeros(int(seconds * rate), np.float32)


# ── VAD ──────────────────────────────────────────────────────────────────────

def test_energy_vad_finds_one_utterance_in_a_stream():
    vad = StreamingVAD(8000, provider="energy", min_speech_ms=250, silence_ms=400)
    stream = np.concatenate([silence(1, 8000), noise(1, 8000), silence(1, 8000)])
    events = []
    for i in range(0, len(stream), 160):               # 20 ms phone packets
        events += vad.feed(stream[i:i + 160])
    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    seconds = len(events[1].audio) / 16_000             # returned at 16 kHz
    assert 1.0 <= seconds <= 1.7                         # speech + pre-roll + a little tail


def test_vad_forces_end_at_max_utterance():
    vad = StreamingVAD(16_000, provider="energy", max_utterance_s=1)
    events = vad.feed(noise(2.5, 16_000))
    assert [e.kind for e in events].count("speech_end") >= 2


def test_silero_vad_loads_and_ignores_silence():
    vad = StreamingVAD(16_000, provider="silero")
    assert vad.provider == "silero"
    assert vad.feed(silence(1, 16_000)) == []


def test_stream_resampler_is_chunk_independent():
    x = np.sin(np.arange(8000) / 7).astype(np.float32)
    whole = _StreamResampler(8000)(x)
    r = _StreamResampler(8000)
    parts = np.concatenate([r(x[i:i + 123]) for i in range(0, len(x), 123)])
    assert abs(len(parts) - len(whole)) <= 1
    n = min(len(parts), len(whole))
    assert np.allclose(parts[:n], whole[:n], atol=1e-5)
    assert abs(len(whole) - 16_000) <= 2
    assert FRAME == 512


# ── audio helpers ────────────────────────────────────────────────────────────

def test_decode_wav_to_phone_pcm():
    wav = encode_wav(np.sin(np.arange(16_000) / 5).astype(np.float32) * 0.5)
    pcm = decode_to_pcm16(wav, 8000)
    assert abs(len(pcm) - 16_000) <= 400                 # 1 s × 8 kHz × 2 bytes


def test_split_frames_meets_exotel_size_rules():
    chunks = split_frames(b"\x01" * 7000, 3200, 320, 3200)
    assert [len(c) for c in chunks] == [3200, 3200, 3200]
    assert chunks[-1].endswith(b"\x00")


# ── Exotel stream end to end ─────────────────────────────────────────────────

class WavTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, language="en"):
        self.calls.append((text, language))
        wav = encode_wav(np.sin(np.arange(8000) / 5).astype(np.float32) * 0.3)   # 0.5 s
        return Speech(base64.b64encode(wav).decode("ascii"), "wav")

    def describe(self):
        return "wav-tts"


def media(samples: np.ndarray) -> dict:
    pcm = (samples * 32767).astype("<i2").tobytes()
    return {"event": "media", "stream_sid": "S1",
            "media": {"chunk": 1, "timestamp": "0", "payload": base64.b64encode(pcm).decode()}}


def read_until_mark(ws) -> list[dict]:
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg["event"] == "mark":
            return msgs


def _call_client(monkeypatch, **overrides):
    s = replace(server.settings, vad_provider="energy", vad_silence_ms=300, **overrides)
    monkeypatch.setattr(server, "settings", s)
    server.engines.stt = FakeSTT()
    server.engines.tts = WavTTS()
    server.engines.llm = make_orchestrator(FakeBackend("fake", tokens=("आज मौसम साफ़ है। ",)))
    return TestClient(server.app)


def test_exotel_call_greets_answers_and_barges_in(monkeypatch):
    client = _call_client(monkeypatch)
    with client.websocket_connect("/telephony/exotel") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json({"event": "start", "stream_sid": "S1", "start": {
            "stream_sid": "S1", "call_sid": "C1", "from": "+919876543210",
            "media_format": {"encoding": "raw", "sample_rate": "8000", "bit_rate": "128"}}})

        greeting = read_until_mark(ws)
        audio = [m for m in greeting if m["event"] == "media"]
        assert audio and all(m["stream_sid"] == "S1" for m in audio)
        sizes = [len(base64.b64decode(m["media"]["payload"])) for m in audio]
        assert all(n >= 3200 and n % 320 == 0 for n in sizes)
        assert server.engines.tts.calls[0][0].startswith("Hello! This is")

        # Caller talks over the greeting → clear; then pauses → a full turn.
        stream = np.concatenate([noise(0.6, 8000), silence(0.6, 8000)])
        for i in range(0, len(stream), 800):
            ws.send_json(media(stream[i:i + 800]))
        turn = read_until_mark(ws)
        events = [m["event"] for m in turn]
        assert events[0] == "clear"
        assert "media" in events and turn[-1]["mark"]["name"] == "turn-2"
        assert server.engines.tts.calls[-1] == ("आज मौसम साफ़ है।", "hi")
        ws.send_json({"event": "stop", "stream_sid": "S1", "stop": {"reason": "callended"}})


def test_exotel_stream_requires_token(monkeypatch):
    client = _call_client(monkeypatch, exotel_ws_token="s3cret", telephony_greeting="off")
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/telephony/exotel?token=wrong") as ws:
            ws.receive_json()
    with client.websocket_connect("/telephony/exotel?token=s3cret") as ws:
        ws.send_json({"event": "stop"})


def test_outbound_call_needs_admin_token(monkeypatch):
    client = _call_client(monkeypatch)
    assert client.post("/telephony/exotel/call", json={"to": "+919876543210"}).status_code == 404
    monkeypatch.setattr(server, "settings", replace(server.settings, admin_token="t"))
    headers = {"Authorization": "Bearer t"}
    assert client.post("/telephony/exotel/call", json={"to": "hello"}, headers=headers).status_code == 422
    resp = client.post("/telephony/exotel/call", json={"to": "+919876543210"}, headers=headers)
    assert resp.status_code == 503 and "EXOTEL_ACCOUNT_SID" in resp.json()["detail"]
