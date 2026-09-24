# VoiceAgent

**A real-time multilingual voice assistant. Speak English or any of nine Indian languages and get a spoken reply. It runs GPU-first on Windows or Linux, with the LLM, STT and TTS providers you choose.**

A browser streams your voice to a FastAPI server over one WebSocket. Speech is transcribed on your NVIDIA GPU with Whisper (or by a cloud STT). The server streams an answer from Gemini, OpenAI, Anthropic Claude or a local Ollama model and speaks it back sentence by sentence while the LLM is still generating.

- **Languages:** English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi
- **Barge-in:** start talking and the bot stops at once; in-flight LLM and TTS work is cancelled
- **Provider failover:** if the primary LLM errors or is slow to start, the next one in the chain answers
- **Live data when it matters:** web search runs only for real-time questions (prices, weather, news); an optional knowledge base comes from Google Discovery Engine
- **Built-in latency telemetry:** per-turn stage timings, a `/metrics` endpoint and a benchmark tool

---

## Contents

1. [Architecture](#architecture)
2. [Requirements](#requirements)
3. [Installation (Windows + NVIDIA GPU)](#installation-windows--nvidia-gpu)
4. [Configuration (.env)](#configuration-env)
5. [Providers and model selection guide](#providers-and-model-selection-guide)
6. [Cost](#cost)
7. [Usage](#usage)
8. [Performance and latency](#performance-and-latency)
9. [Benchmarking](#benchmarking)
10. [Production deployment](#production-deployment)
11. [Troubleshooting](#troubleshooting)
12. [Project structure](#project-structure)

---

## Architecture

```
Browser (mic) ── 16 kHz PCM WAV over WebSocket ──►  FastAPI /ws
                                                        │
                    ┌───────────────────────────────────┘
                    ▼
        STT  (faster-whisper on CUDA │ Sarvam Saaras │ OpenAI │ Indic-Seamless)
          Silero VAD trims silence · language ID restricted to STT_LANGUAGES
                    │ text + language
                    ▼
        Language ID (script-based, Lingua for Hindi vs Marathi)
                    │
                    ▼
        Retrieval planner ── conversational / general → none (answer immediately)
                    │       └ real-time (price, weather, news…) → web search race
                    │       └ KB configured → Discovery Engine ∥ web, first quality hit
                    │  (waits at most LLM_RETRIEVAL_WAIT; results cached)
                    ▼
        LLM chain  (Gemini → OpenAI → Anthropic → Ollama, configurable)
          streaming · first-token timeout · failover · cool-down · bounded concurrency
                    │ text deltas
                    ▼
        Speech chunker: first clause ASAP, then whole sentences
                    │
                    ▼
        TTS router  (Sarvam Bulbul │ ElevenLabs Flash │ OpenAI │ Edge fallback)
          concurrent synthesis · strict in-order delivery · LRU cache
                    │ base64 MP3 chunks
                    ▼
Browser (speaker) — plays each chunk as it arrives
```

Each user turn runs as its own `asyncio.Task`, so a barge-in cancels it immediately. All network I/O is async over pooled keep-alive connections. Local GPU inference runs in a worker thread, so the event loop never blocks.

---

## Requirements

| Component | Requirement |
|---|---|
| OS | Windows 10/11 64-bit (primary) or Linux x86-64 |
| Python | 3.11 or 3.12 (64-bit) |
| GPU (recommended) | NVIDIA GPU with ≥ 6 GB VRAM for `large-v3-turbo` Whisper; ≥ 12 GB if you also run an Ollama LLM locally |
| NVIDIA driver | R550 or newer (CUDA 12.x capable). **No CUDA Toolkit install is needed**: the cuBLAS and cuDNN 9 runtimes come from pip wheels (`requirements-gpu.txt`) |
| CPU-only | Supported (Whisper `small`, int8). Slower STT, or use a cloud STT (`sarvam` / `openai`) |
| Ollama (optional) | [ollama.com/download](https://ollama.com/download) for fully local LLMs. It uses the NVIDIA GPU automatically |
| Browser | Chrome or Edge (desktop or Android). Microphone access needs `http://localhost` or HTTPS |

---

## Installation (Windows + NVIDIA GPU)

All commands are for **PowerShell**. On Linux, use `source .venv/bin/activate` and `cp` instead of `Copy-Item`.

```powershell
# 1. Get the code
git clone <repo-url> VoiceAgent
cd VoiceAgent

# 2. Virtual environment (Python 3.12)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-gpu.txt    # NVIDIA CUDA 12 runtime (cuBLAS + cuDNN 9) for local Whisper

# Optional extras
pip install -r requirements-kb.txt        # Google Discovery Engine knowledge base
pip install -r requirements-seamless.txt  # AI4Bharat Indic-Seamless STT (PyTorch CUDA, ~3 GB)

# 4. Verify the GPU is visible to Whisper (prints the number of CUDA devices; 0 = CPU fallback)
python -c "from config.device import cuda_device_count; print(cuda_device_count())"

# 5. Configure
Copy-Item .env.example .env
notepad .env                           # add at least one LLM API key (or use Ollama)

# 6. (Optional) local LLM
ollama pull gemma3:12b                 # 8 GB GPUs: gemma3:4b

# 7. Run
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, click **Start Listening** and speak. On first start, Whisper downloads its weights once (about 1.6 GB for `large-v3-turbo`) into the Hugging Face cache.

Check that everything loaded:

```powershell
curl http://localhost:8000/health
# {"status":"ok","engines":{"stt":"faster-whisper:large-v3-turbo@cuda","llm":"gemini:gemini-3.5-flash-lite → openai:gpt-6-luna","tts":"sarvam+edge"}}
```

---

## Configuration (.env)

Everything is configured through environment variables (`.env`), and nothing is hard-coded. **Never commit `.env`.** It is already in `.gitignore`. `.env.example` documents every variable. These are the ones that matter most:

### API keys (set only what you use)

| Variable | Used for | Get it |
|---|---|---|
| `GEMINI_API_KEY` | Gemini LLM | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| `OPENAI_API_KEY` | OpenAI LLM / STT / TTS | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Claude LLM | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `SARVAM_API_KEY` | Indic TTS (Bulbul) / STT (Saaras) | [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS | [elevenlabs.io](https://elevenlabs.io/app/settings/api-keys) |
| `BRAVE_SEARCH_API_KEY` | Production web search | [api-dashboard.search.brave.com](https://api-dashboard.search.brave.com) |

### Provider selection

| Variable | Values | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini` · `openai` · `anthropic` · `ollama` | `gemini` |
| `LLM_FALLBACKS` | comma list, tried in order | *(none)* |
| `STT_PROVIDER` | `whisper` · `sarvam` · `openai` · `seamless` | `whisper` |
| `TTS_PROVIDER` | `auto` · `sarvam` · `elevenlabs` · `openai` · `edge` | `auto` |
| `RETRIEVAL_MODE` | `auto` · `always` · `off` | `auto` |

Providers without a key are skipped with a warning. Edge TTS needs no key and is always the last TTS fallback.

### Models and tuning

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_MODEL` / `GEMINI_THINKING_LEVEL` | `gemini-3.5-flash-lite` / `minimal` | 3.8 Flash accepts `low`+ only; an unsupported level falls back to the model default automatically |
| `OPENAI_MODEL` / `OPENAI_REASONING_EFFORT` | `gpt-6-luna` / `none` | `none` = fastest first token |
| `OPENAI_BASE_URL` | — | Any OpenAI-compatible endpoint (Azure OpenAI v1, vLLM, LM Studio) |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | For `claude-sonnet-5`, set `ANTHROPIC_THINKING=disabled` (and optionally `ANTHROPIC_EFFORT=low`) for voice latency |
| `OLLAMA_MODEL` / `OLLAMA_KEEP_ALIVE` | `gemma3:12b` / `30m` | keep-alive keeps weights resident in VRAM |
| `WHISPER_MODEL` | *(auto)* | `large-v3-turbo` on GPU, `small` on CPU; also `medium`, `large-v3`, `distil-large-v3` (English) |
| `WHISPER_COMPUTE_TYPE` | `auto` | `float16` on GPU, `int8` on CPU; `int8_float16` saves VRAM |
| `STT_LANGUAGE` | *(empty)* | Force one language (e.g. `hi`) to skip language detection |
| `STT_LANGUAGES` | all 10 | Whisper's language ID is restricted to these (Urdu → Hindi, etc.) |
| `LLM_MAX_TOKENS` | `400` | Spoken replies are short; caps cost and runaway answers |
| `LLM_FIRST_TOKEN_TIMEOUT` | `6` s | Fail over if no token arrives in time |
| `LLM_RETRIEVAL_WAIT` | `1.2` s | Longest the answer waits for web/KB context |
| `TTS_FIRST_CHUNK_MIN_CHARS` / `_MAX_CHARS` | `24` / `70` | How early the first audio is cut |
| `VAD_SILENCE_MS` | `550` | Browser end-of-speech pause (sent to the client via `/config`) |

---

## Providers and model selection guide

These recommendations were researched in September 2026 against each provider's current documentation. They optimize for **voice**: time-to-first-token (TTFT) matters more than peak intelligence, because every 100 ms is audible. The defaults in `.env.example` follow the ★ rows.

### Conversation (the spoken reply) — the latency-critical path

| | Model | Why |
|---|---|---|
| ★ Default | **`gemini-3.5-flash-lite`** (Gemini, `GEMINI_THINKING_LEVEL=minimal`) | Google's fastest, lowest-cost Gemini tier; minimal thinking keeps TTFT low; strong Indic coverage |
| ★ Fallback | **`gpt-6-luna`** (OpenAI, `reasoning_effort=none`) | Cost-efficient GPT-6 tier built for fast everyday work ($0.10 / $0.50 per 1M in/out tokens) |
| Alternative | **`claude-haiku-4-5`** (Anthropic) | Anthropic's fastest model; no extended thinking by default ($1 / $5 per 1M) |
| Offline | **`gemma3:12b`** / `gemma3:4b` (Ollama) | Good multilingual quality that runs on a single consumer GPU; `gemma4` / `qwen3` tags also work |

### Harder questions (reasoning, multi-step, careful answers)

| Model | Notes |
|---|---|
| `gemini-3.8-flash` (`GEMINI_THINKING_LEVEL=low`) | Google's most capable Flash model; noticeably slower first token than Flash-Lite |
| `gpt-6-sol` | OpenAI's faster reasoning tier; keep `reasoning_effort` at `none` or `low` for voice |
| `claude-sonnet-5` (`ANTHROPIC_THINKING=disabled`, `ANTHROPIC_EFFORT=low`) | High quality at $2 / $10 per 1M; tune thinking/effort for latency |
| `gpt-6-astra`, `claude-opus-5`, `claude-fable-5-1` | Frontier reasoning, but **too slow for real-time voice**; use them for offline or back-office tasks |

### Other task types (if you extend the assistant)

| Task | Recommended | Notes |
|---|---|---|
| Intent / routing classification | Regex rules (built in, 0 ms) → `gemini-3.5-flash-lite` or `gpt-6-luna` | Don't spend an LLM call on the hot path when a rule suffices; this project routes retrieval with multilingual regexes |
| Entity extraction (structured JSON) | `gpt-6-luna`, `gemini-3.5-flash-lite`, `claude-haiku-4-5` with structured outputs / JSON schema | Small models are accurate enough for extraction and 5–20× cheaper |
| Embeddings (custom RAG) | `gemini-embedding-001` (text) or Gemini Embedding 2 (multimodal); `text-embedding-3-small` (cost) / `-large` (quality) | Not needed by default: the KB uses Discovery Engine's managed retrieval |
| Speech-to-text, local GPU | ★ **faster-whisper `large-v3-turbo`** (float16) | Free, private, near large-v3 accuracy at a fraction of the compute |
| Speech-to-text, Indian languages / Hinglish | **Sarvam `saaras:v3`** | Best Indic and code-mixed accuracy; cloud, ≤ 30 s per request |
| Speech-to-text, cloud multilingual | OpenAI `gpt-4o-mini-transcribe` (or `gpt-transcribe` for accuracy) | Simple REST |
| Text-to-speech, Indian languages | ★ **Sarvam `bulbul:v3`** | Most natural Indic voices; handles Hinglish and numbers |
| Text-to-speech, lowest latency (English, Hindi, Tamil) | **ElevenLabs `eleven_flash_v2_5`** | ~75 ms model latency (vendor figure) |
| Text-to-speech, expressive | OpenAI `gpt-4o-mini-tts` | Steerable tone via instructions |
| Text-to-speech, free | Edge neural voices | No key needed; about 0.6 s fixed connection overhead per sentence (measured) |
| Web search | **Brave Search API** | Official API; scrapers (DuckDuckGo + Bing, raced) are the free fallback |

> **Speech-to-speech models** (OpenAI `gpt-realtime-2.1` / `gpt-live-1`, Gemini `gemini-3.8-live`) can go lower still by merging STT, LLM and TTS into one model. They are not wired in here, because the pipeline design keeps per-language voice control (Sarvam), retrieval and provider independence. They are the natural next step if English-first latency is all that matters.

Model names change often. Check the provider's model page before upgrading, and change the `*_MODEL` variables. No code changes are needed.

---

## Cost

List prices were checked in **September 2026** on each provider's pricing page (standard, pay-as-you-go tier, USD unless noted). Prices change often, so confirm them before budgeting. INR prices are converted at **₹88 = $1**.

### How a "turn" is estimated

Per-turn costs below assume one typical voice exchange:

| Quantity | Assumption | Why |
|---|---|---|
| User speech | 5 s of audio | A short spoken question |
| LLM input | 1,200 tokens | System prompt (~300) + `HISTORY_TURNS=8` of short history (~850) + question (~50). Add ~1,000 when web/KB context is injected |
| LLM output | 80 tokens | 1–4 spoken sentences; hard cap is `LLM_MAX_TOKENS=400` |
| Spoken reply | 350 characters ≈ 20 s of audio | 80 tokens of text |
| Web search | 10 % of turns | `RETRIEVAL_MODE=auto` only searches for real-time questions |

### LLM (per 1M tokens)

| Model | Provider | Input | Output | Per 1,000 turns |
|---|---|---|---|---|
| ★ `gemini-3.5-flash-lite` | Google | $0.30 | $2.50 | **$0.56** |
| ★ `gpt-6-luna` | OpenAI | $0.10 | $0.50 | **$0.16** |
| `claude-haiku-4-5` | Anthropic | $1.00 | $5.00 | $1.60 |
| `gemini-3.8-flash` | Google | $0.75 → $1.50 from 1 Jan 2027 | $3.75 → $7.50 from 1 Jan 2027 | $1.20 (→ $2.40) |
| `gpt-6-sol` | OpenAI | $2.00 | $10.00 | $3.20 |
| `claude-sonnet-5` | Anthropic | $2.00 | $10.00 | $3.20 |
| `claude-opus-5` | Anthropic | $5.00 | $25.00 | $8.00 |
| `gpt-6-astra` | OpenAI | $10.00 | $50.00 | $16.00 |
| `claude-fable-5-1` | Anthropic | $10.00 | $50.00 | $16.00 |
| `gemma3:12b` / `gemma3:4b` (Ollama) | local | $0 | $0 | GPU cost only (see [local GPU](#local-models-on-a-gpu)) |

Thinking/reasoning tokens are billed as output. Keep `GEMINI_THINKING_LEVEL=minimal`, `OPENAI_REASONING_EFFORT=none` and `ANTHROPIC_THINKING=disabled` (Sonnet 5), or the output figures above can rise several-fold.

### Speech-to-text

| Model | Provider | Price | Per 1,000 turns (5 s each) |
|---|---|---|---|
| ★ faster-whisper `large-v3-turbo` | local | $0 | GPU cost only |
| AI4Bharat Indic-Seamless | local | $0 | GPU cost only |
| `gpt-4o-mini-transcribe` | OpenAI | $0.003 / min ($0.18 / hour) | $0.25 |
| `gpt-transcribe` | OpenAI | $0.0045 / min ($0.27 / hour) | $0.38 |
| `saaras:v3` | Sarvam | ₹30 / hour (≈ $0.34 / hour) | ₹42 ≈ $0.47 |

### Text-to-speech

| Model | Provider | Price | Per 1,000 turns (350 chars each) |
|---|---|---|---|
| Edge neural voices | Microsoft (unofficial endpoint) | Free, no SLA | **$0** |
| `gpt-4o-mini-tts` | OpenAI | $0.60 / 1M text tokens + $12 / 1M audio tokens (≈ $0.015 / min) | ≈ $5.00 |
| ★ `bulbul:v3` | Sarvam | ₹3 / 1,000 characters (≈ $0.034) | ₹1,050 ≈ $11.90 |
| `eleven_flash_v2_5` | ElevenLabs | $0.05 / 1,000 characters | $17.50 |

**TTS is usually the largest cost in the pipeline**: a paid voice costs 10–100× more per turn than a fast LLM. The ElevenLabs figure is the API usage rate; subscription plans bundle characters differently.

### Retrieval

| Service | Price | Per 1,000 turns |
|---|---|---|
| DuckDuckGo + Bing scraping (fallback) | Free, best-effort | $0 |
| Brave Search API | $5 / 1,000 requests ($5 of monthly credit included) | ≈ $0.50 at 10 % of turns |
| Google Discovery Engine (KB) | Per-query and storage pricing on [Google Cloud](https://cloud.google.com/generative-ai-app-builder/pricing) | Depends on data store size |

### Example stacks (per 1,000 turns)

| Stack | STT | LLM | TTS | Search | **Total** | ≈ per turn |
|---|---|---|---|---|---|---|
| Default, English (Whisper on GPU + Flash-Lite + Edge) | GPU | $0.56 | $0 | $0.50 | **$1.06** + GPU | $0.001 |
| Default, Indic (Whisper on GPU + Flash-Lite + Sarvam) | GPU | $0.56 | $11.90 | $0.50 | **$12.96** + GPU | $0.013 |
| All-cloud, cheapest (OpenAI mini STT + Luna + Edge) | $0.25 | $0.16 | $0 | $0.50 | **$0.91** | $0.001 |
| All-cloud, Indic quality (Saaras + Flash-Lite + Bulbul) | $0.47 | $0.56 | $11.90 | $0.50 | **$13.43** | $0.013 |
| Low-latency English (OpenAI mini STT + Haiku + ElevenLabs Flash) | $0.25 | $1.60 | $17.50 | $0.50 | **$19.85** | $0.020 |
| Fully local (Whisper + Ollama `gemma3:12b` + Edge) | GPU | GPU | $0 | $0 (scrapers) | **GPU only** | — |

For scale: 10,000 conversations a month of 10 turns each is 100,000 turns. That costs about $106 on the default English stack or about $1,300 on the default Indic stack, plus the GPU.

### Local models on a GPU

Local models (Whisper, Indic-Seamless, Ollama) have no per-token fee; you pay for the GPU instead. These are **estimates, not measurements** (GPU throughput was not benchmarked for this README). Run `python scripts/benchmark.py components` on your hardware to replace them.

**What fits on one GPU**

| GPU | VRAM | Runs well |
|---|---|---|
| RTX 3060 / 4060 (8–12 GB) | 8–12 GB | Whisper `large-v3-turbo` (~3 GB) + `gemma3:4b` (~4 GB) |
| RTX 4090 / L4 / L40S | 24–48 GB | Whisper `large-v3-turbo` + `gemma3:12b` (~9 GB) with room for parallel requests |

**Hourly GPU prices (September 2026, on-demand)**

| GPU | Where | Price / hour | 24 × 7 per month (730 h) |
|---|---|---|---|
| RTX 4090 24 GB | Marketplace clouds (Salad, RunPod community, …) | $0.33 cheapest · $0.44 median | $240 – $320 |
| RTX 4090 24 GB | RunPod secure cloud | ~$0.69 | ~$500 |
| L4 24 GB | AWS `g6.xlarge` (us-east-1) | $0.80 | ~$590 |
| L40S 48 GB | Marketplace clouds | from ~$0.55 | from ~$400 |
| Your own RTX 4090 | Electricity at $0.15 / kWh, ~400 W under load | ~$0.06 while busy | ~$20–45 + hardware (~$1,800 ÷ 36 months ≈ $50) |

**Estimated cost per 1,000 turns on a GPU you rent by the hour**

Assumed GPU time per turn: Whisper `large-v3-turbo` ≈ 0.3 s for a 5 s clip on an RTX 4090 (≈ 0.6 s on an L4); `gemma3:12b` ≈ 2 s on an RTX 4090 or ≈ 4 s on an L4 (prefill + 80 tokens). Voice traffic is bursty, so the table assumes the GPU is busy **50 %** of the hours you pay for.

| Workload | GPU | Turns / hour at 50 % busy | Per 1,000 turns |
|---|---|---|---|
| Whisper STT only | RTX 4090 ($0.44 / h) | ~6,000 | ≈ $0.07 |
| Whisper STT only | L4 ($0.80 / h) | ~3,000 | ≈ $0.27 |
| Whisper + `gemma3:12b` | RTX 4090 ($0.44 / h) | ~780 | ≈ $0.56 |
| Whisper + `gemma3:12b` | L4 ($0.80 / h) | ~390 | ≈ $2.05 |
| Whisper + `gemma3:12b` | Your own RTX 4090 (electricity only) | — | ≈ $0.04 |

**Rules of thumb**

- A rented GPU is billed whether it is busy or idle. At low traffic, cloud STT + LLM (≈ $0.41–1.03 per 1,000 turns) is cheaper than keeping a GPU running. A $0.44/h RTX 4090 left on 24 × 7 (~$320/month) only pays for itself above roughly **300,000–800,000 turns a month**.
- Local wins on **privacy** (audio never leaves your machine), **no rate limits** and **predictable latency**. On hardware you already own, the marginal cost is almost only electricity.
- Local LLMs replace only the cheap part of the bill. TTS is the largest cost, and this project has no local TTS, so Edge (free) is the zero-cost voice.

---

## Usage

**Voice:** click **Start Listening** and speak. After a pause of `VAD_SILENCE_MS`, the utterance is sent. Talk over the bot to interrupt it.

**Text:** type into the chat box. The reply is streamed and spoken the same way.

### WebSocket protocol (`/ws`)

| Client → server | Meaning |
|---|---|
| binary frame | 16-bit PCM WAV utterance (mono, 16 kHz preferred) |
| `{"type":"text","text":"..."}` | typed message |
| `{"type":"interrupt"}` | barge-in: cancel the current reply |
| `{"type":"audio","audio":"<base64 WAV>"}` | legacy clients |

| Server → client | Meaning |
|---|---|
| `transcription` | what the user said |
| `status` (`thinking`) | a reply has started |
| `chunk` | streamed reply text |
| `audio_chunk` | `{audio: base64, format: "mp3", text}`, played in order |
| `done` | `{text, source: "none"|"web"|"kb"}` |
| `metrics` | `{timings_ms: {stt, retrieval_done, llm_ttft, first_audio, total}}` |
| `error` / `interrupt` | error message / acknowledgement |

### HTTP endpoints

`GET /health` (engines loaded) · `GET /metrics` (latency percentiles) · `GET /config` (client settings) · `GET /ping`

---

## Performance and latency

### Where the time goes (one voice turn)

```
end of speech ─► VAD_SILENCE_MS ─► STT ─► [retrieval] ─► LLM first token ─► first clause ─► TTS ─► audio
                   550 ms           GPU:    0 ms for      provider TTFT      ~0.5 s at      provider
                   (browser)        fast    most turns                       50 tok/s
```

### What this version optimizes

- **Retrieval off the hot path.** General questions no longer wait on a web search. Only real-time queries (and KB queries, if configured) retrieve. Web engines run concurrently, results are cached, and concurrent identical queries share one fetch.
- **Earlier first audio.** The first TTS segment is cut at the first clause boundary after 24 characters (or at a word boundary at 70), instead of at the end of the first sentence or a fixed 90 characters. The emotion-tag prefix the old prompt forced before every answer is gone.
- **Streaming everywhere.** LLM tokens stream to the client and to TTS. TTS runs concurrently per segment and is delivered in order.
- **Connection reuse.** One pooled HTTP client (HTTP/2 when available) for Sarvam, ElevenLabs and scraping. SDK clients are singletons. Warmup at startup opens TLS connections and loads model weights. Ollama keeps weights resident (`OLLAMA_KEEP_ALIVE`).
- **Failover without penalty.** First-token timeout plus a 20–30 s cool-down means a failing provider costs one slow turn, not every turn.
- **Bounded concurrency.** Separate semaphores for LLM streams and TTS requests, and a lock serializing GPU STT, so load cannot trigger rate-limit storms or GPU thrashing.
- **Leaner STT.** No ffmpeg or pydub decode (raw PCM is parsed with numpy). Silero VAD runs on ONNX (no PyTorch). Greedy decoding, no timestamps, bounded tokens, and language ID restricted to supported languages.
- **Parallel startup.** STT model load, LLM warmup, TTS warmup and language models all load concurrently.

### Tuning checklist

| Goal | Setting |
|---|---|
| Lowest STT latency on GPU | `WHISPER_MODEL=large-v3-turbo` (default); `WHISPER_COMPUTE_TYPE=int8_float16` if VRAM is tight; `STT_LANGUAGE=hi` if all users speak one language |
| Lowest LLM TTFT | Flash-Lite / Luna / Haiku with minimal thinking (defaults); keep `LLM_MAX_TOKENS` small |
| Faster end-of-turn | Lower `VAD_SILENCE_MS` (450–500) in quiet environments |
| Faster first audio | ElevenLabs Flash (English/Hindi/Tamil) or Sarvam (Indic) instead of Edge; lower `TTS_FIRST_CHUNK_MIN_CHARS` |
| Real-time answers without waiting | `RETRIEVAL_MODE=auto` (default) and `BRAVE_SEARCH_API_KEY` |
| Never wait on retrieval | `RETRIEVAL_MODE=off` |

---

## Benchmarking

Live per-turn timings are logged for every reply, sent to the browser (`metrics` messages; see the browser console) and aggregated at `GET /metrics` (response shape; the numbers are illustrative):

```json
{"stt": {"count": 42, "p50_ms": 180.2, "p95_ms": 260.9, "max_ms": 301.0},
 "llm_ttft": {...}, "first_audio": {...}, "total": {...}}
```

All values are milliseconds since the user's audio (or text) reached the server.

The benchmark tool (`scripts/benchmark.py`) measures real providers using your `.env`:

```powershell
# Each configured provider in isolation: LLM TTFT per model, TTS per engine/language,
# STT on synthetic English + Hindi utterances, web search
python scripts/benchmark.py components --runs 5

# End-to-end against a running server (client-side: transcription, first text, first audio)
python scripts/benchmark.py e2e --url ws://localhost:8000/ws --runs 5 [--audio my_question.wav]
```

Results print as tables and are saved to `bench_results/*.json`.

### Before vs after (measured on the development machine)

These were measured on a CPU-only Windows laptop with no GPU and no LLM API keys. The old code (git `HEAD`) and the new code ran on identical inputs. The LLM was an instant stub, so orchestration overhead is isolated; provider TTFT is identical in both versions and is excluded.

| Stage | Before | After |
|---|---|---|
| Overhead before the LLM starts, general question (no live data needed) | ~500 ms (fixed retrieval wait) | **~0 ms** |
| Overhead before the LLM starts, real-time question | 365–505 ms, **weather query answered without data** | 127–260 ms, **live data for 3/3 queries** |
| Mean across 10 mixed queries | 433 ms | **55 ms** |
| Web search for a real-time query | 2.1–4.5 s (DuckDuckGo first, sequential cascade) | **~0.3 s** (engines raced) |
| First TTS segment ready, English reply at 50 tok/s | 793 ms (cut mid-word at 90 chars) | **538 ms** (word boundary) |
| First TTS segment ready, Hindi reply | 1463 ms | **721 ms** |
| Whisper `base` on CPU (int8), 4–5 s clip | 1.6–1.8 s | 1.6–1.8 s (unchanged; GPU not measured here) |
| Edge TTS, first sentence | ~0.9 s | ~0.9 s (fixed service overhead; use Sarvam or ElevenLabs to reduce) |

**Estimated time to first audio (excluding STT and LLM TTFT, which are unchanged): ~2.2 s → ~1.5 s for an English general question, and ~2.9 s → ~1.7 s in Hindi.** Run `scripts/benchmark.py` on your GPU machine with your keys to get numbers for your setup.

---

## Production deployment

- **Run one worker per GPU.** Models live in-process, so extra uvicorn workers would load extra Whisper copies. Scale out with more instances behind a load balancer that supports WebSockets (sessions are per connection; no shared state is needed).
  ```powershell
  uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --ws-ping-interval 20 --timeout-keep-alive 30
  ```
- **HTTPS is required** for microphone access anywhere except `localhost`. Put a reverse proxy in front (Caddy, Nginx, IIS with the WebSocket module) or use `cloudflared tunnel --url http://localhost:8000` for quick sharing.
- **Lock down CORS:** `CORS_ORIGINS=https://your.domain`.
- **Windows service:** run under [NSSM](https://nssm.cc) or Task Scheduler with the venv's `python.exe -m uvicorn ...` and the project folder as working directory.
- **Secrets:** inject API keys as environment variables from your secret store; `.env` is for development.
- **Health and monitoring:** `/health` returns 503 until all engines are loaded (use it for readiness probes). Scrape `/metrics` for latency SLOs. Logs rotate in `logs/voiceagent.log` (10 MB × 5).
- **Rate limits:** tune `LLM_MAX_CONCURRENCY` and `TTS_MAX_CONCURRENCY` to your provider tiers. SDK and HTTP retries honor `Retry-After` with short back-off, then fail over.
- **Web search:** use `BRAVE_SEARCH_API_KEY` in production. HTML scraping is best-effort and subject to the search engines' terms.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` shows `@cpu` although you have an NVIDIA GPU | `pip install -r requirements-gpu.txt`, update the NVIDIA driver (R550+), then run the verify command from Installation. Force it with `WHISPER_DEVICE=cuda` to see the actual CUDA error |
| `Could not load library cudnn_ops64_9.dll` / `cublas64_12.dll` | Same as above: the cuDNN 9 / cuBLAS 12 wheels are missing. They are registered automatically from the venv |
| `WHISPER_MODEL=mlx-community/...` error | That is an Apple MLX model; use `large-v3-turbo`, `medium` or `small` |
| CUDA out of memory | `WHISPER_MODEL=medium` or `WHISPER_COMPUTE_TYPE=int8_float16`; use a smaller Ollama model |
| `No usable LLM provider` at startup | Set `LLM_PROVIDER` and its API key, or `LLM_PROVIDER=ollama` with Ollama running |
| Every reply fails with "temporarily unavailable" | All providers in the chain failed; the log shows why (bad key, model name, network). Try `python scripts/benchmark.py components` |
| Ollama: connection refused | Start Ollama (tray app or `ollama serve`) and `ollama pull <model>`; check `OLLAMA_HOST` |
| Hindi transcribed in Urdu script | Use `large-v3-turbo` (small models confuse the two), set `STT_LANGUAGE=hi`, or use `STT_PROVIDER=sarvam`. The reply is in Hindi either way |
| Microphone does not work | Allow mic permission; use `http://localhost` or HTTPS |
| No audio output | Click the page once (browsers need a user gesture before audio plays) |
| Answers ignore live data | Check the query is real-time (prices, weather, news); raise `LLM_RETRIEVAL_WAIT`; add `BRAVE_SEARCH_API_KEY` |
| Knowledge base errors | `pip install -r requirements-kb.txt`; `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`; check `GCP_PROJECT_ID` / `GCP_DATA_STORE_ID` |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

Set `LOG_LEVEL=DEBUG` for detailed logs.

---

## Project structure

```
VoiceAgent/
├── server.py                 FastAPI app: WebSocket pipeline, streaming TTS, /health /metrics /config
├── config/
│   ├── settings.py           All environment configuration (single source of truth)
│   ├── device.py             CUDA detection + Windows cuBLAS/cuDNN DLL registration
│   └── logging_config.py     Console + rotating file logging (Windows-safe)
├── core/
│   ├── chunker.py            Streaming text → speakable segments
│   ├── lang.py               Script-based language ID, speech text cleanup
│   ├── http.py               Shared async HTTP client + retry/429 handling
│   └── metrics.py            Per-turn timers and rolling percentiles
├── llm/
│   ├── orchestrator.py       History, retrieval gating, failover, concurrency
│   ├── providers/            gemini.py · openai_llm.py · anthropic_llm.py · ollama_llm.py
│   ├── retrieval.py          Multilingual routing, KB + web race, cache
│   └── web_search.py         Brave API · DuckDuckGo + Bing race · gold-rate scraper
├── stt/
│   ├── faster_whisper_stt.py Local Whisper (CUDA float16 / CPU int8)
│   ├── cloud_stt.py          Sarvam Saaras · OpenAI transcription
│   ├── seamless.py           AI4Bharat Indic-Seamless (optional)
│   └── audio.py              WAV parsing, resampling, VAD trimming
├── tts/
│   ├── __init__.py           TTS router: per-language chain, fallback, cache
│   ├── sarvam.py · cloud_tts.py (OpenAI, ElevenLabs) · edge_tts_engine.py
├── static/                   index.html + app.js (browser client)
├── scripts/benchmark.py      Component and end-to-end latency benchmarks
├── tests/                    Offline unit and WebSocket tests (python -m pytest)
├── requirements*.txt         core · gpu · kb · seamless
└── .env.example
```

### Development

```powershell
pip install pytest
python -m pytest -q                       # offline: no GPU, network or keys required
uvicorn server:app --reload --port 8000
```

To add a provider, implement `llm.base.LLMBackend`, `tts.base.TTSBase` or `stt.base.STTBase`, then register it in the matching factory (`llm/providers/__init__.py`, `tts/__init__.py`, `stt/__init__.py`).
