"""
server.py
VoiceAgent — FastAPI server.

Pipeline per user turn (one WebSocket per browser tab):
  browser PCM WAV ─► STT ─► language ID ─► [knowledge base → Google, when needed]
                 ─► LLM stream (+ booking actions) ─► speech chunker ─► TTS ─► browser

  • Every heavy engine is created and warmed up in parallel at startup.
  • Each turn runs in its own asyncio.Task so barge-in cancels it instantly,
    including TTS requests that are still in flight.
  • Per-turn stage timings are logged, sent to the client and aggregated at /metrics.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from config.logging_config import get_logger, setup_logging  # noqa: E402
from config.settings import get_settings  # noqa: E402

setup_logging()
logger = get_logger("server")

from core import lang  # noqa: E402
from core.chunker import SpeechChunker  # noqa: E402
from core.http import close_http_client  # noqa: E402
from core.metrics import STATS, TurnTimer  # noqa: E402
from core.privacy import mask_pii  # noqa: E402
from llm import AllProvidersFailed, LLMOrchestrator  # noqa: E402
from stt import STTBase, build_stt_engine  # noqa: E402
from stt.audio import parse_wav  # noqa: E402
from tts import TTSRouter, build_tts_router  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
settings = get_settings()


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
            "agent_name": settings.agent_name}


@app.get("/bookings", include_in_schema=False)
async def list_bookings(date: str | None = None, authorization: str = Header(default="")):
    """Bookings for staff. Disabled unless ADMIN_TOKEN is set; send
    'Authorization: Bearer <ADMIN_TOKEN>'. Contains customer phone numbers."""
    token = settings.admin_token
    if not token or not engines.llm:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    if not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await asyncio.to_thread(engines.llm.bookings.list_bookings, date)


# ────────────────────────────────────────────────────────────────────────────
# STREAMING TTS — ordered delivery, concurrent synthesis
# ────────────────────────────────────────────────────────────────────────────

class StreamingTTS:
    """
    LLM text → SpeechChunker → one TTS task per segment (started immediately,
    concurrency bounded by the router) → sent to the client strictly in order.
    """

    def __init__(self, ws: WebSocket, language: str, timer: TurnTimer) -> None:
        self.ws, self.language, self.timer = ws, language, timer
        self.chunker = SpeechChunker(settings.tts_first_chunk_min_chars,
                                     settings.tts_first_chunk_max_chars)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending: list[asyncio.Task] = []
        self._sender = asyncio.create_task(self._send_loop())

    def _dispatch(self, segments: list[str]) -> None:
        for text in segments:
            task = asyncio.create_task(engines.tts.synthesize(text, self.language))
            self._pending.append(task)
            self._queue.put_nowait((text, task))

    def feed(self, text: str) -> None:
        self._dispatch(self.chunker.feed(text))

    async def finish(self) -> None:
        self._dispatch(self.chunker.flush())
        self._queue.put_nowait(None)
        await self._sender

    def cancel(self) -> None:
        for task in (*self._pending, self._sender):
            task.cancel()

    async def _send_loop(self) -> None:
        while (item := await self._queue.get()) is not None:
            text, task = item
            try:
                speech = await task
            except Exception as exc:
                logger.error("TTS failed for %.40s: %r", text, exc)
                continue
            if speech is None:
                continue
            self.timer.mark("first_audio")
            await self.ws.send_json({"type": "audio_chunk", "audio": speech.audio_b64,
                                     "format": speech.fmt, "text": text})


# ────────────────────────────────────────────────────────────────────────────
# WEBSOCKET
# ────────────────────────────────────────────────────────────────────────────

_MAX_FRAME_BYTES = settings.max_audio_seconds * 16_000 * 2 * 2 + 44   # allow 32 kHz input


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session_id = uuid.uuid4().hex[:12]
    current: asyncio.Task | None = None
    logger.info("Client connected | session=%s", session_id)

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
        tts = StreamingTTS(ws, language, timer)
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

    async def transcribe(samples, sample_rate: int, timer: TurnTimer) -> tuple[str, str]:
        result = await engines.stt.transcribe(samples, sample_rate)
        timer.mark("stt")
        text = result["text"].strip()
        return text, lang.detect_language(text, result.get("language"))

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            audio_bytes, text_raw = msg.get("bytes"), msg.get("text")
            timer = TurnTimer()
            user_text, language = "", "en"

            if audio_bytes:
                await cancel_current(notify=True)
                if len(audio_bytes) > _MAX_FRAME_BYTES:
                    await ws.send_json({"type": "error", "text": "Audio too long — please keep it under "
                                        f"{settings.max_audio_seconds} seconds."})
                    continue
                try:
                    samples, sr = parse_wav(audio_bytes)
                    user_text, language = await transcribe(samples, sr, timer)
                except Exception as exc:
                    logger.error("STT error: %r", exc)
                    await send_error("Sorry, I couldn't process that audio.")
                    continue
                if not user_text:
                    continue
                await ws.send_json({"type": "transcription", "text": user_text})

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
                await cancel_current(notify=True)
                if kind == "audio":   # legacy base64-WAV clients
                    try:
                        samples, sr = parse_wav(base64.b64decode(payload.get("audio") or ""))
                        user_text, language = await transcribe(samples, sr, timer)
                    except Exception as exc:
                        logger.error("STT error (base64): %r", exc)
                        await send_error("Sorry, I couldn't process that audio.")
                        continue
                    if not user_text:
                        continue
                    await ws.send_json({"type": "transcription", "text": user_text})
                else:
                    user_text = str(payload.get("text", "")).strip()[:2000]
                    if not user_text:
                        continue
                    language = lang.detect_language(user_text)
            else:
                continue

            logger.info("USER [%s|%s]: %.120s", session_id, language, mask_pii(user_text))
            current = asyncio.create_task(respond(user_text, language, timer))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("WebSocket error | session=%s | %r", session_id, exc)
    finally:
        if current and not current.done():
            current.cancel()
        if engines.llm:
            engines.llm.release_session(session_id)
        logger.info("Client disconnected | session=%s", session_id)
