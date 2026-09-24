"""
scripts/benchmark.py — latency benchmarks for VoiceAgent.

  python scripts/benchmark.py components [--runs 5]
      Measures every provider configured in .env, in isolation:
        LLM time-to-first-token (TTFT) and total time, per backend in the chain
        TTS latency for a typical first sentence, per engine and language
        STT latency on synthetic Hindi + English utterances
        Web search latency (real-time query)

  python scripts/benchmark.py e2e [--url ws://localhost:8000/ws] [--runs 5] [--audio file.wav]
      Drives a running server over WebSocket, like the browser does, and records
      client-side time to transcription, first text and first audio.

Results print as a table and are saved to bench_results/<mode>-<timestamp>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

PROMPTS = [
    ("en", "Explain in two sentences how solar panels generate electricity."),
    ("hi", "सौर पैनल बिजली कैसे बनाते हैं? दो वाक्यों में बताइए।"),
]
TTS_SENTENCES = {
    "en": "Solar panels turn sunlight into electricity using photovoltaic cells.",
    "hi": "सौर पैनल सूरज की रोशनी को बिजली में बदलते हैं।",
}


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {}
    s = sorted(samples)
    return {"n": len(s), "p50_ms": round(statistics.median(s)),
            "p95_ms": round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]),
            "min_ms": round(s[0])}


def print_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"  {'name':42s} {'p50':>8s} {'p95':>8s} {'min':>8s} {'n':>4s}")
    for name, st in rows.items():
        if st:
            print(f"  {name:42s} {st['p50_ms']:>6d}ms {st['p95_ms']:>6d}ms {st['min_ms']:>6d}ms {st['n']:>4d}")
        else:
            print(f"  {name:42s}   failed")


# ── components ────────────────────────────────────────────────────────────────

async def bench_llm(runs: int) -> dict:
    from config.settings import get_settings
    from llm.providers import build_backend
    s = get_settings()
    out: dict[str, dict] = {}
    for name in s.llm_chain():
        backend = build_backend(name, s)
        if not backend.available:
            continue
        try:
            await backend.warmup()
        except Exception as exc:
            print(f"  {backend.describe()}: warmup failed ({exc!r})")
            out[f"{backend.describe()} ttft"] = {}
            continue
        for lang, prompt in PROMPTS:
            ttft, total = [], []
            for _ in range(runs):
                t0 = time.perf_counter()
                first = None
                try:
                    async for _chunk in backend.stream("You are a concise voice assistant.",
                                                       [{"role": "user", "content": prompt}]):
                        first = first or time.perf_counter()
                except Exception as exc:
                    print(f"  {backend.describe()} error: {exc!r}")
                    break
                if first:
                    ttft.append((first - t0) * 1000)
                    total.append((time.perf_counter() - t0) * 1000)
            out[f"{backend.describe()} [{lang}] ttft"] = summarize(ttft)
            out[f"{backend.describe()} [{lang}] total"] = summarize(total)
        await backend.close()
    return out


async def bench_tts(runs: int) -> dict:
    from config.settings import get_settings
    from tts import _build
    s = get_settings()
    out: dict[str, dict] = {}
    for name in ("edge", "sarvam", "openai", "elevenlabs"):
        engine = _build(name, s)
        if not engine.available:
            continue
        await engine.warmup()
        for lang, text in TTS_SENTENCES.items():
            if not engine.supports(lang):
                continue
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                if await engine.synthesize(text, lang):
                    times.append((time.perf_counter() - t0) * 1000)
            out[f"{name} [{lang}]"] = summarize(times)
        await engine.close()
    return out


async def synth_sample(lang: str) -> "tuple":
    """Synthetic test utterance via Edge TTS, decoded to 16 kHz float32."""
    import io

    import edge_tts
    from faster_whisper import decode_audio
    voice = {"en": "en-IN-PrabhatNeural", "hi": "hi-IN-MadhurNeural"}[lang]
    text = {"en": "What is the weather like in Mumbai today?",
            "hi": "आज मुंबई में मौसम कैसा रहेगा?"}[lang]
    buf = io.BytesIO()
    async for ch in edge_tts.Communicate(text, voice).stream():
        if ch["type"] == "audio":
            buf.write(ch["data"])
    buf.seek(0)
    return decode_audio(buf, sampling_rate=16000), text


async def bench_stt(runs: int) -> dict:
    from config.settings import get_settings
    from stt import build_stt_engine
    s = get_settings()
    engine = await asyncio.to_thread(build_stt_engine, s)
    out: dict[str, dict] = {}
    for lang in ("en", "hi"):
        samples, _ = await synth_sample(lang)
        await engine.transcribe(samples, 16000)   # warm
        times, text = [], ""
        for _ in range(runs):
            t0 = time.perf_counter()
            text = (await engine.transcribe(samples, 16000))["text"]
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  STT [{lang}] → {text!r}")
        out[f"{engine.describe()} [{lang}]"] = summarize(times)
    await engine.close()
    return out


async def bench_web(runs: int) -> dict:
    from llm.web_search import WebSearchService
    w = WebSearchService()
    await w.warmup()
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        if await w.search(f"weather in Mumbai today {i}"):
            times.append((time.perf_counter() - t0) * 1000)
    return {"web search (real-time query)": summarize(times)}


async def run_components(runs: int) -> dict:
    results = {}
    for title, fn in (("LLM", bench_llm), ("TTS", bench_tts), ("STT", bench_stt), ("Web search", bench_web)):
        try:
            results[title] = await fn(runs)
        except Exception as exc:
            print(f"{title} benchmark failed: {exc!r}")
            results[title] = {}
        print_table(title, results[title])
    from core.http import close_http_client
    await close_http_client()
    return results


# ── end-to-end over WebSocket ─────────────────────────────────────────────────

async def run_e2e(url: str, runs: int, audio_path: str | None) -> dict:
    import websockets  # installed with uvicorn[standard]
    turns: list[tuple[str, object]] = [("text", t) for _, t in PROMPTS]
    if audio_path:
        turns.append(("audio", Path(audio_path).read_bytes()))
    else:
        from stt.audio import encode_wav
        samples, _ = await synth_sample("en")
        turns.append(("audio", encode_wav(samples)))

    stats: dict[str, list[float]] = {}
    server_side: list[dict] = []
    for kind, payload in turns:
        for _ in range(runs):
            async with websockets.connect(url, max_size=None) as ws:
                t0 = time.perf_counter()
                if kind == "text":
                    await ws.send(json.dumps({"type": "text", "text": payload}))
                else:
                    await ws.send(payload)
                marks: dict[str, float] = {}
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                    now = (time.perf_counter() - t0) * 1000
                    key = {"transcription": "transcription", "chunk": "first_text",
                           "audio_chunk": "first_audio", "done": "done"}.get(msg["type"])
                    if key and key not in marks:
                        marks[key] = now
                    if msg["type"] == "metrics":
                        server_side.append(msg["timings_ms"])
                        break
                    if msg["type"] == "error":
                        print("  server error:", msg.get("text"))
                        break
                for k, v in marks.items():
                    stats.setdefault(f"{kind}: {k}", []).append(v)
    table = {k: summarize(v) for k, v in stats.items()}
    print_table(f"End-to-end (client-side, {url})", table)
    return {"client": table, "server_timings": server_side}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["components", "e2e"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--audio", help="16-bit PCM WAV to send in e2e mode")
    args = ap.parse_args()

    if args.mode == "components":
        results = asyncio.run(run_components(args.runs))
    else:
        results = asyncio.run(run_e2e(args.url, args.runs, args.audio))

    out_dir = ROOT / "bench_results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
