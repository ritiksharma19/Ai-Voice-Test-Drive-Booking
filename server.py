"""
server.py
VoiceAgent — FastAPI server.

Pipeline per user turn (one WebSocket per browser tab or phone call):
  browser PCM WAV / Exotel PCM stream + server VAD ─► STT ─► language ID ─► [knowledge base → Google, when needed]
                 ─► LLM stream (+ booking actions) ─► speech chunker ─► TTS ─► browser

  • Every heavy engine is created and warmed up in parallel at startup.
  • Each turn (STT included) runs in its own asyncio.Task so barge-in cancels it
    instantly, including TTS requests that are still in flight.
  • Browser sessions: optional ACCESS_TOKEN, MAX_SESSIONS cap, MAX_TURNS_PER_MINUTE.
  • Phone calls (Exotel) arrive at /telephony/exotel; see telephony/exotel.py.
  • Per-turn stage timings are logged, sent to the client and aggregated at /metrics.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import re
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from config.logging_config import get_logger, setup_logging  # noqa: E402
from config.settings import get_settings  # noqa: E402

setup_logging()
logger = get_logger("server")

from core import lang, vad  # noqa: E402
from core.http import close_http_client  # noqa: E402
from core.metrics import STATS, TurnTimer  # noqa: E402
from core.privacy import indian_mobile, mask_pii  # noqa: E402
from core.speech_stream import StreamingTTS  # noqa: E402
from llm import AllProvidersFailed, LLMOrchestrator  # noqa: E402
from llm.knowledge_base import (  # noqa: E402
    delete_document, list_documents, resolve_kb_dir, save_document)
from stt import STTBase, build_stt_engine  # noqa: E402
from stt.audio import parse_wav  # noqa: E402
from telephony import exotel  # noqa: E402
from telephony.callback import CallbackLimiter  # noqa: E402
from tts import TTSRouter, build_tts_router  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
settings = get_settings()
callbacks = CallbackLimiter(settings)


class Engines:
    stt: STTBase | None = None
    llm: LLMOrchestrator | None = None
    tts: TTSRouter | None = None


engines = Engines()


# ────────────────────────────────────────────────────────────────────────────
# LIFESPAN — build and warm every engine concurrently
# ────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 VoiceAgent starting — loading engines in parallel…")
    engines.llm = LLMOrchestrator(settings)
    engines.tts = build_tts_router(settings)
    results = await asyncio.gather(
        asyncio.to_thread(build_stt_engine, settings),   # model load is blocking
        engines.llm.warmup(),
        engines.tts.warmup(),
        asyncio.to_thread(lang.warmup),
        asyncio.to_thread(vad.warmup, settings.vad_provider),   # phone-call VAD
        return_exceptions=True,
    )
    if isinstance(results[0], BaseException):
        raise RuntimeError(f"STT engine failed to load: {results[0]}") from results[0]
    engines.stt = results[0]
    logger.info("✅ Ready | stt=%s | llm=%s | tts=%s", engines.stt.describe(),
                " → ".join(b.describe() for b in engines.llm.backends), engines.tts.describe())
    yield
    logger.info("Shutting down…")
    await asyncio.gather(engines.llm.close(), engines.tts.close(), engines.stt.close(),
                         close_http_client(), return_exceptions=True)


app = FastAPI(title="VoiceAgent", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.get("/health")
async def health():
    ready = all((engines.stt, engines.llm, engines.tts))
    return JSONResponse(status_code=200 if ready else 503, content={
        "status": "ok" if ready else "starting",
        "engines": {
            "stt": engines.stt.describe() if engines.stt else None,
            "llm": " → ".join(b.describe() for b in engines.llm.backends) if engines.llm else None,
            "tts": engines.tts.describe() if engines.tts else None,
        },
    })


@app.get("/metrics")
async def metrics():
    """Rolling latency percentiles per pipeline stage (ms since end of user speech)."""
    return STATS.snapshot()


@app.get("/config")
async def client_config():
    return {"vad_silence_ms": settings.vad_silence_ms,
            "business_name": settings.business_name,
            "agent_name": settings.agent_name,
            "callback_enabled": settings.callback_enabled,     # show the "Call me back" button
            "callback_ready": _callbacks_available(),         # Exotel outbound is configured
            "kb_uploads": bool(engines.llm and engines.llm.retrieval.uses_local_kb)}


@app.get("/bookings", include_in_schema=False)
async def list_bookings(date: str | None = None, authorization: str = Header(default="")):
    """Bookings for staff. Disabled unless ADMIN_TOKEN is set; send
    'Authorization: Bearer <ADMIN_TOKEN>'. Contains customer phone numbers."""
    if denied := _staff_denied(authorization):
        return denied
    return await asyncio.to_thread(engines.llm.bookings.list_bookings, date)


def _staff_denied(authorization: str) -> JSONResponse | None:
    """404 unless ADMIN_TOKEN is set; 401 unless 'Authorization: Bearer <ADMIN_TOKEN>'."""
    token = settings.admin_token
    if not token or not engines.llm:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    if not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return None


def _page_denied(token: str) -> JSONResponse | None:
    """401 unless the page's ACCESS_TOKEN is supplied as ?token= (no ACCESS_TOKEN = open)."""
    if settings.access_token and not hmac.compare_digest(token.encode(), settings.access_token.encode()):
        return JSONResponse(status_code=401, content={"detail": "Access token required"})
    return None


# ────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE DOCUMENTS — upload .md / .txt / .pdf from the page
# Open to anyone who can use the page (ACCESS_TOKEN when set), like chat itself.
# ────────────────────────────────────────────────────────────────────────────

def _kb_root() -> Path:
    return resolve_kb_dir(engines.llm.retrieval.s.kb_dir)


def _kb_unavailable() -> JSONResponse | None:
    if not engines.llm.retrieval.uses_local_kb:
        return JSONResponse(status_code=409, content={
            "detail": "Uploads feed the local knowledge base, but KB_PROVIDER resolves to "
                      f"'{engines.llm.retrieval.s.kb_backend()}'."})
    return None


@app.get("/kb/documents", include_in_schema=False)
async def kb_documents(token: str = ""):
    if denied := _page_denied(token):
        return denied
    return {"documents": await asyncio.to_thread(list_documents, _kb_root()),
            "uploads_enabled": engines.llm.retrieval.uses_local_kb,
            "max_mb": settings.kb_upload_max_mb}


@app.put("/kb/documents/{filename}", include_in_schema=False)
async def kb_upload(filename: str, request: Request, token: str = ""):
    """Raw file bytes as the body. Re-uploading a name replaces that document."""
    if denied := _page_denied(token) or _kb_unavailable():
        return denied
    limit = settings.kb_upload_max_mb * 1024 * 1024
    data = bytearray()
    async for part in request.stream():
        data += part
        if len(data) > limit:
            return JSONResponse(status_code=413, content={
                "detail": f"File is larger than {settings.kb_upload_max_mb} MB"})
    try:
        stored = await asyncio.to_thread(save_document, _kb_root(), filename, bytes(data))
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    chunks = await asyncio.to_thread(engines.llm.retrieval.reload_local_kb)
    logger.info("KB document uploaded: %s (%d bytes) → %d chunks", stored, len(data), chunks)
    return {"stored_as": stored, "chunks": chunks}


@app.delete("/kb/documents/{name}", include_in_schema=False)
async def kb_delete(name: str, token: str = ""):
    """Only uploaded documents (KB_DIR/uploads) can be deleted."""
    if denied := _page_denied(token) or _kb_unavailable():
        return denied
    if not await asyncio.to_thread(delete_document, _kb_root(), name):
        return JSONResponse(status_code=404, content={"detail": "No uploaded document with that name"})
    chunks = await asyncio.to_thread(engines.llm.retrieval.reload_local_kb)
    logger.info("KB document deleted: %s → %d chunks", name, chunks)
    return {"deleted": name, "chunks": chunks}


# ────────────────────────────────────────────────────────────────────────────
# CALL ME BACK — the AI agent phones the customer (Exotel outbound)
# ────────────────────────────────────────────────────────────────────────────

def _callbacks_available() -> bool:
    return settings.callback_enabled and settings.outbound_calls_configured


def _client_id(request: Request) -> str:
    # Behind Cloudflare or a reverse proxy the socket peer is the proxy. These headers
    # can be forged without one, but the per-number and hourly limits still apply.
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return (request.headers.get("cf-connecting-ip") or forwarded
            or (request.client.host if request.client else "unknown"))


@app.post("/callback", include_in_schema=False)
async def request_callback(body: dict, request: Request, token: str = ""):
    """Customer asks for a call: {"phone": "98765 43210"}. Public, so it is guarded by
    ACCESS_TOKEN (when set, as ?token=) and the CallbackLimiter limits."""
    if denied := _page_denied(token):
        return denied
    if not _callbacks_available():
        return JSONResponse(status_code=404, content={"detail": "Call-backs are not available"})
    number = indian_mobile(body.get("phone", ""))
    if not number:
        return JSONResponse(status_code=422, content={
            "detail": "Please enter a 10-digit Indian mobile number."})
    client = _client_id(request)
    if reason := callbacks.check(number, client):
        logger.warning("Call-back refused for %s: %s", mask_pii(number), reason)
        return JSONResponse(status_code=429, content={"detail": reason})
    try:
        call = await exotel.place_call(settings, number)
    except Exception as exc:
        callbacks.release(number, client)
        logger.error("Call-back to %s failed: %r", mask_pii(number), exc)
        return JSONResponse(status_code=502, content={
            "detail": "We couldn't place the call. Please try again in a moment."})
    logger.info("Call-back placed to %s | call_sid=%s", mask_pii(number), call.get("call_sid"))
    return {"status": "calling"}


# ────────────────────────────────────────────────────────────────────────────
# TELEPHONY — Exotel Voicebot applet (bidirectional stream)
# ────────────────────────────────────────────────────────────────────────────

@app.websocket("/telephony/exotel")
async def exotel_stream(ws: WebSocket) -> None:
    if not exotel.authorized(ws, settings.exotel_ws_token):
        logger.warning("Exotel stream rejected: bad or missing token")
        await ws.close(code=1008)
        return
    await ws.accept()
    await exotel.ExotelCall(ws, engines, settings).run()


@app.post("/telephony/exotel/call", include_in_schema=False)
async def exotel_outbound_call(body: dict, authorization: str = Header(default="")):
    """Staff only: {"to": "+919876543210"} rings the customer and connects them
    to the agent through the Exotel call flow EXOTEL_APP_ID."""
    if denied := _staff_denied(authorization):
        return denied
    to = re.sub(r"[\s\-()]", "", str(body.get("to", "")))
    if not re.fullmatch(r"\+?\d{10,15}", to):
        return JSONResponse(status_code=422, content={"detail": "'to' must be a phone number"})
    try:
        return await exotel.place_call(settings, to)
    except ValueError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except Exception as exc:
        logger.error("Exotel outbound call failed: %r", exc)
        return JSONResponse(status_code=502, content={"detail": "Exotel request failed"})


# ────────────────────────────────────────────────────────────────────────────
# WEBSOCKET
# ────────────────────────────────────────────────────────────────────────────

_MAX_FRAME_BYTES = settings.max_audio_seconds * 16_000 * 2 * 2 + 44   # allow 32 kHz input
_active_sessions = 0


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    global _active_sessions
    await ws.accept()
    # Accept first so the browser sees the close code: 1008 = bad token, 1013 = full.
    if not exotel.authorized(ws, settings.access_token):
        logger.warning("Browser session rejected: bad or missing ACCESS_TOKEN")
        await ws.close(code=1008, reason="Access token required")
        return
    if settings.max_sessions and _active_sessions >= settings.max_sessions:
        logger.warning("Browser session rejected: MAX_SESSIONS=%d reached", settings.max_sessions)
        await ws.close(code=1013, reason="Server busy")
        return
    _active_sessions += 1
    session_id = uuid.uuid4().hex[:12]
    current: asyncio.Task | None = None
    turn_times: deque[float] = deque()
    logger.info("Client connected | session=%s | active=%d", session_id, _active_sessions)

    def rate_limited() -> bool:
        """Sliding one-minute window of turns (MAX_TURNS_PER_MINUTE, 0 = off)."""
        limit = settings.max_turns_per_minute
        if limit <= 0:
            return False
        now = time.monotonic()
        while turn_times and now - turn_times[0] > 60:
            turn_times.popleft()
        if len(turn_times) >= limit:
            return True
        turn_times.append(now)
        return False

    async def cancel_current(notify: bool) -> None:
        nonlocal current
        if current and not current.done():
            current.cancel()
            try:
                await current
            except (asyncio.CancelledError, Exception):
                pass
            if notify:
                await ws.send_json({"type": "interrupt"})
        current = None

    async def send_error(text: str) -> None:
        try:
            await ws.send_json({"type": "error", "text": text})
        except Exception:
            pass   # socket already closed

    async def respond(user_text: str, language: str, timer: TurnTimer) -> None:
        await ws.send_json({"type": "status", "status": "thinking"})

        async def emit(text: str, speech) -> None:
            await ws.send_json({"type": "audio_chunk", "audio": speech.audio_b64,
                                "format": speech.fmt, "text": text})

        tts = StreamingTTS(engines.tts, emit, language, timer, settings)
        reply: list[str] = []
        source = "none"
        try:
            async for chunk in engines.llm.stream_reply(user_text, session_id, language, timer):
                if chunk.get("booking"):
                    await ws.send_json({"type": "booking", "booking": chunk["booking"]})
                    continue
                if chunk["provider"] != "filler":
                    source = chunk["source"]
                reply.append(chunk["text"])
                await ws.send_json({"type": "chunk", "text": chunk["text"]})
                tts.feed(chunk["text"])
            await tts.finish()
        except asyncio.CancelledError:
            tts.cancel()
            logger.info("Turn cancelled (barge-in) | session=%s", session_id)
            raise
        except AllProvidersFailed as exc:
            tts.cancel()
            logger.error("All LLM providers failed: %s", exc)
            await send_error("The assistant is temporarily unavailable. Please try again.")
            return
        except Exception as exc:
            tts.cancel()
            logger.exception("Turn failed: %r", exc)
            await send_error("Something went wrong. Please try again.")
            return

        timer.mark("total")
        logger.info("BOT [%s] src=%s | %s | %.100s", language, source, timer.summary(),
                    mask_pii("".join(reply)))
        await ws.send_json({"type": "done", "text": "".join(reply), "audio": None, "source": source})
        await ws.send_json({"type": "metrics", "timings_ms": timer.summary()})

    async def turn(timer: TurnTimer, wav: bytes | None = None, text: str = "") -> None:
        """One user turn: STT (for audio), then the reply. Runs as a task so the
        receive loop stays free for barge-in while Whisper is busy."""
        if wav is not None:
            try:
                samples, sr = parse_wav(wav)
                result = await engines.stt.transcribe(samples, sr)
            except Exception as exc:
                logger.error("STT error: %r", exc)
                await send_error("Sorry, I couldn't process that audio.")
                return
            timer.mark("stt")
            text = result["text"].strip()
            if not text:
                return
            language = lang.detect_language(text, result.get("language"))
            await ws.send_json({"type": "transcription", "text": text})
        else:
            language = lang.detect_language(text)
        logger.info("USER [%s|%s]: %.120s", session_id, language, mask_pii(text))
        await respond(text, language, timer)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            audio_bytes, text_raw = msg.get("bytes"), msg.get("text")
            wav: bytes | None = None
            user_text = ""

            if audio_bytes:
                wav = audio_bytes
            elif text_raw:
                try:
                    payload = json.loads(text_raw)
                except json.JSONDecodeError:
                    continue
                kind = payload.get("type", "text")
                if kind == "interrupt":
                    logger.info("Barge-in | session=%s", session_id)
                    await cancel_current(notify=False)
                    await ws.send_json({"type": "interrupt"})
                    continue
                if kind == "audio":   # legacy base64-WAV clients
                    try:
                        wav = base64.b64decode(payload.get("audio") or "")
                    except ValueError:
                        await send_error("Sorry, I couldn't process that audio.")
                        continue
                else:
                    user_text = str(payload.get("text", "")).strip()[:2000]
                    if not user_text:
                        continue
            else:
                continue

            await cancel_current(notify=True)
            if wav is not None and len(wav) > _MAX_FRAME_BYTES:
                await send_error("Audio too long — please keep it under "
                                 f"{settings.max_audio_seconds} seconds.")
                continue
            if rate_limited():
                logger.warning("Rate limit reached | session=%s", session_id)
                await send_error("You're sending messages too quickly. Please wait a moment.")
                continue
            current = asyncio.create_task(turn(TurnTimer(), wav, user_text))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("WebSocket error | session=%s | %r", session_id, exc)
    finally:
        _active_sessions -= 1
        if current and not current.done():
            current.cancel()
        if engines.llm:
            engines.llm.release_session(session_id)
        logger.info("Client disconnected | session=%s", session_id)
