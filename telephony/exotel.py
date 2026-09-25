"""
telephony/exotel.py
Phone calls through Exotel, over the Voicebot applet's bidirectional stream.

  caller ─► Exotel ─► wss://<host>/telephony/exotel   (JSON events, base64 16-bit PCM)
     media ─► StreamingVAD ─► speech_end ─► STT ─► LLM ─► TTS (MP3)
                                          ─► PCM at the call's rate ─► media ─► caller
     speech_start while the agent talks ─► cancel the turn + "clear" (barge-in)

Exotel → server: connected · start · media · dtmf · mark · stop
Server → Exotel: media (≥ 3.2 KB, a multiple of 320 bytes) · mark · clear

Outbound calls use the Calls/connect API: Exotel rings the customer, then
connects them to your call flow (whose Voicebot applet points here).
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import time
import uuid

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from config.logging_config import get_logger
from config.settings import Settings
from core import lang
from core.http import get_http_client
from core.metrics import TurnTimer
from core.privacy import mask_pii
from core.speech_stream import StreamingTTS
from core.vad import StreamingVAD, VADEvent
from llm import AllProvidersFailed
from stt.audio import SAMPLE_RATE
from telephony.audio import BYTES_PER_SAMPLE, decode_to_pcm16, pcm16_to_float, split_frames
from tts.base import Speech

logger = get_logger("telephony.exotel")

_MIN_CHUNK = 3200     # Exotel: outbound media ≥ 3.2 KB …
_ALIGN = 320          # … and a multiple of 320 bytes
_SORRY = "Sorry, I'm having trouble right now. Please try again in a moment."


def authorized(ws: WebSocket, token: str) -> bool:
    """EXOTEL_WS_TOKEN as `?token=` or as the Basic-auth password
    (wss://user:<token>@host/telephony/exotel). No token configured = open."""
    if not token:
        return True
    supplied = ws.query_params.get("token", "")
    auth = ws.headers.get("authorization", "")
    if not supplied and auth.lower().startswith("basic "):
        try:
            supplied = base64.b64decode(auth[6:]).decode().partition(":")[2]
        except Exception:
            return False
    return bool(supplied) and hmac.compare_digest(supplied.encode(), token.encode())


def greeting(s: Settings) -> str:
    return s.telephony_greeting or (
        f"Hello! This is {s.agent_name} from {s.business_name}. How can I help you today?")


class ExotelCall:
    """One phone call: VAD turn-taking, barge-in and the STT → LLM → TTS pipeline."""

    def __init__(self, ws: WebSocket, engines, settings: Settings) -> None:
        self.ws, self.engines, self.s = ws, engines, settings
        self.stream_sid = ""
        self.session_id = f"exotel-{uuid.uuid4().hex[:12]}"
        self.sample_rate = 8000
        self.vad: StreamingVAD | None = None
        self.current: asyncio.Task | None = None
        self._chunk_bytes = _MIN_CHUNK
        self._playback_until = 0.0          # estimated end of audio queued at Exotel
        self._last_mark = ""
        self._marks = 0
        self._turn_audio: np.ndarray | None = None
        self._spoke = False                 # has the current turn sent audio yet?
        self._carry: np.ndarray | None = None
        self._send_lock = asyncio.Lock()

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            while True:
                try:
                    msg = json.loads(await self.ws.receive_text())
                except json.JSONDecodeError:
                    continue
                event = msg.get("event")
                if event == "media":
                    await self._on_media(msg.get("media") or {})
                elif event == "start":
                    await self._on_start(msg)
                elif event == "mark":
                    if (msg.get("mark") or {}).get("name") == self._last_mark:
                        self._playback_until = 0.0      # Exotel finished playing our audio
                elif event == "dtmf":
                    logger.info("DTMF | session=%s", self.session_id)
                elif event == "stop":
                    logger.info("Call ended | session=%s | %s", self.session_id,
                                (msg.get("stop") or {}).get("reason", ""))
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("Exotel stream error | session=%s | %r", self.session_id, exc)
        finally:
            if self.current and not self.current.done():
                self.current.cancel()
            if self.engines.llm:
                self.engines.llm.release_session(self.session_id)
            logger.info("Call disconnected | session=%s", self.session_id)

    async def _on_start(self, msg: dict) -> None:
        start = msg.get("start") or {}
        self.stream_sid = msg.get("stream_sid") or start.get("stream_sid", "")
        if start.get("call_sid"):
            self.session_id = f"exotel-{start['call_sid']}"
        try:
            self.sample_rate = int((start.get("media_format") or {}).get("sample_rate") or 8000)
        except (TypeError, ValueError):
            self.sample_rate = 8000
        # ~100 ms per media message, within Exotel's size rules
        chunk = max(_MIN_CHUNK, self.sample_rate * BYTES_PER_SAMPLE // 10)
        self._chunk_bytes = chunk + (-chunk % _ALIGN)
        self.vad = StreamingVAD.from_settings(self.s, self.sample_rate)
        logger.info("Call started | session=%s | from=%s | %d Hz | vad=%s", self.session_id,
                    mask_pii(str(start.get("from", ""))), self.sample_rate, self.vad.provider)
        if self.s.telephony_greeting.lower() != "off":
            self._start_task(self.say(greeting(self.s), self.s.telephony_language))

    async def _on_media(self, media: dict) -> None:
        if self.vad is None or not media.get("payload"):
            return
        for event in self.vad.feed(pcm16_to_float(base64.b64decode(media["payload"]))):
            await self._on_vad(event)

    async def _on_vad(self, event: VADEvent) -> None:
        if event.kind == "speech_start":
            if self.current and not self.current.done():
                # A turn still thinking was cut short by a pause: keep its audio
                # and transcribe it together with what the caller says next.
                carry = None if self._spoke else self._turn_audio
                await self._cancel_current()
                self._carry = carry
            if time.monotonic() < self._playback_until:
                logger.info("Barge-in | session=%s", self.session_id)
                await self._send({"event": "clear", "stream_sid": self.stream_sid})
                self._playback_until = 0.0
        elif event.kind == "speech_end" and event.audio is not None:
            audio = event.audio
            if self._carry is not None:
                audio, self._carry = np.concatenate([self._carry, audio]), None
            self._start_task(self._turn(audio))

    # ── turns ─────────────────────────────────────────────────────────────────

    def _start_task(self, coro) -> None:
        if self.current and not self.current.done():
            self.current.cancel()
        self._turn_audio, self._spoke = None, False
        self.current = asyncio.create_task(coro)

    async def _cancel_current(self) -> None:
        if self.current and not self.current.done():
            self.current.cancel()
            try:
                await self.current
            except (asyncio.CancelledError, Exception):
                pass
        self.current = None

    async def _turn(self, audio: np.ndarray) -> None:
        self._turn_audio = audio
        timer = TurnTimer()
        try:
            result = await self.engines.stt.transcribe(audio, SAMPLE_RATE)
        except Exception as exc:
            logger.error("STT error | session=%s | %r", self.session_id, exc)
            return
        timer.mark("stt")
        text = result["text"].strip()
        if not text:
            return
        language = lang.detect_language(text, result.get("language"))
        logger.info("CALLER [%s|%s]: %.120s", self.session_id, language, mask_pii(text))

        stream = StreamingTTS(self.engines.tts, self._play, language, timer, self.s)
        reply: list[str] = []
        try:
            async for chunk in self.engines.llm.stream_reply(text, self.session_id, language, timer):
                if chunk.get("booking"):
                    logger.info("Booking %s confirmed on call | session=%s",
                                chunk["booking"].get("booking_id"), self.session_id)
                    continue
                reply.append(chunk["text"])
                stream.feed(chunk["text"])
            await stream.finish()
        except asyncio.CancelledError:
            stream.cancel()
            raise
        except Exception as exc:
            stream.cancel()
            if isinstance(exc, AllProvidersFailed):
                logger.error("All LLM providers failed: %s", exc)
            else:
                logger.exception("Call turn failed: %r", exc)
            await self.say(_SORRY, "en")
            return
        timer.mark("total")
        logger.info("BOT [%s] %s | %.100s", language, timer.summary(), mask_pii("".join(reply)))
        await self._send_mark()

    async def say(self, text: str, language: str) -> None:
        speech = await self.engines.tts.synthesize(text, language)
        if speech is not None:
            await self._play(text, speech)
            await self._send_mark()

    # ── audio out ─────────────────────────────────────────────────────────────

    async def _play(self, text: str, speech: Speech) -> None:
        try:
            pcm = await asyncio.to_thread(decode_to_pcm16, base64.b64decode(speech.audio_b64),
                                          self.sample_rate)
        except Exception as exc:
            logger.error("Could not decode TTS audio (%s) for %.40s: %r", speech.fmt, text, exc)
            return
        if not pcm:
            return
        self._spoke = True
        for chunk in split_frames(pcm, self._chunk_bytes, _ALIGN, _MIN_CHUNK):
            await self._send({"event": "media", "stream_sid": self.stream_sid,
                              "media": {"payload": base64.b64encode(chunk).decode("ascii")}})
        seconds = len(pcm) / (self.sample_rate * BYTES_PER_SAMPLE)
        self._playback_until = max(time.monotonic(), self._playback_until) + seconds

    async def _send_mark(self) -> None:
        self._marks += 1
        self._last_mark = f"turn-{self._marks}"
        await self._send({"event": "mark", "stream_sid": self.stream_sid,
                          "mark": {"name": self._last_mark}})

    async def _send(self, payload: dict) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload))


# ── outbound calls ───────────────────────────────────────────────────────────

async def place_call(s: Settings, to: str) -> dict:
    """Ring `to`, then connect them to the call flow EXOTEL_APP_ID (Calls/connect API).
    Not retried: a retry after a timeout could ring the customer twice."""
    required = {"EXOTEL_ACCOUNT_SID": s.exotel_account_sid, "EXOTEL_API_KEY": s.exotel_api_key,
                "EXOTEL_API_TOKEN": s.exotel_api_token, "EXOTEL_CALLER_ID": s.exotel_caller_id,
                "EXOTEL_APP_ID": s.exotel_app_id}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Outbound calls need {', '.join(missing)}")
    sid = s.exotel_account_sid
    resp = await get_http_client().post(
        f"https://{s.exotel_subdomain}/v1/Accounts/{sid}/Calls/connect.json",
        auth=(s.exotel_api_key, s.exotel_api_token),
        data={"From": to, "CallerId": s.exotel_caller_id, "CallType": "trans",
              "Url": f"http://my.exotel.com/{sid}/exoml/start_voice/{s.exotel_app_id}"},
        timeout=15.0,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Exotel HTTP {resp.status_code}: {resp.text[:200]}")
    call = resp.json().get("Call", {})
    return {"call_sid": call.get("Sid"), "status": call.get("Status")}
