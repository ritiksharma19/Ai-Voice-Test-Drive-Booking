// ==========================================================================
// VoiceAgent — Client JavaScript
// ==========================================================================

// ── WebSocket ──────────────────────────────────────────────────────────────
let ws;
const host = window.location.host || "127.0.0.1:8000";
const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
// ACCESS_TOKEN on the server → open the page as /?token=<ACCESS_TOKEN>
const accessToken = new URLSearchParams(window.location.search).get("token");
const socketUrl = `${protocol}//${host}/ws` + (accessToken ? `?token=${encodeURIComponent(accessToken)}` : "");

// Exponential backoff reconnect state
let _wsReconnectDelay = 1000;
const _WS_RECONNECT_MAX = 30000;

// ── App state ──────────────────────────────────────────────────────────────
let appState = "idle";  // idle | listening | thinking | speaking
let isSpeaking = false;
let isStreamFinished = false;
let currentAssistantBubble = null;
let currentAssistantText = "";
let audioQueue = [];
let isPlayingAudioQueue = false;
let selectedVoiceName = "";
let businessLabel = "VoiceAgent";   // replaced by "<agent> · <business>" from /config

// ── Listening / recording ──────────────────────────────────────────────────
let isListeningEnabled = false;
let scriptProcessor = null;   // ScriptProcessorNode — replaces MediaRecorder
let pcmBuffer = [];     // raw Float32 samples at 16 kHz (accumulated)
let micStream = null;   // raw MediaStream from getUserMedia
let audioContext = null;
let analyser = null;
let micSource = null;
let silenceTimer = null;
let silenceMs = 550;           // overridden by /config
let speechDetected = false;
let animationFrameId = null;

// ── Barge-in tracking ──────────────────────────────────────────────────────
let speakingStartTime = 0;   // timestamp when bot started speaking
let bargedIn = false; // prevent double-interrupt in same speaking turn

// ── Audio playback ─────────────────────────────────────────────────────────
let activeAudio = null;

// ── Canvas waveform ────────────────────────────────────────────────────────
let waveCanvas = null;
let waveCtx = null;
let waveAnimId = null;

// ── DOM refs (set in init) ─────────────────────────────────────────────────
let connectionStatus, stateTag, statePrompt, orb, orbIcon,
    toggleMicBtn, clearBtn, chatContainer, voiceSelect, chatForm, chatInput;

// ==========================================================================
// INIT
// ==========================================================================

let audioUnlocked = false;

function unlockAudio() {
    if (audioUnlocked) return;
    const s = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
    s.play().then(() => { audioUnlocked = true; }).catch(() => { });
}

function init() {
    connectionStatus = document.getElementById("connection-status");
    stateTag = document.getElementById("state-tag");
    statePrompt = document.getElementById("state-prompt");
    orb = document.getElementById("orb");
    orbIcon = document.getElementById("orb-icon");
    toggleMicBtn = document.getElementById("toggle-mic");
    clearBtn = document.getElementById("clear-btn");
    chatContainer = document.getElementById("chat-container");
    voiceSelect = document.getElementById("voice-select");
    chatForm = document.getElementById("chat-form");
    chatInput = document.getElementById("chat-input");

    initWaveformCanvas();
    setupWebSocket();
    setupSpeechRecognition();
    setupEventListeners();
    setupCallback();
    setupKnowledgeBase();
    fetchSystemInfo();
    fetchClientConfig();

    document.body.addEventListener('click', unlockAudio, { once: true });
    document.body.addEventListener('touchstart', unlockAudio, { once: true });
    document.body.addEventListener('keydown', unlockAudio, { once: true });
}

// ==========================================================================
// SYSTEM INFO
// ==========================================================================

async function fetchClientConfig() {
    try {
        const res = await fetch('/config');
        if (!res.ok) return;
        const cfg = await res.json();
        if (Number.isFinite(cfg.vad_silence_ms)) silenceMs = cfg.vad_silence_ms;
        if (cfg.agent_name && cfg.business_name) {
            businessLabel = `${cfg.agent_name} · ${cfg.business_name}`;
            const h = document.querySelector("#welcome-msg h3");
            if (h) h.textContent = `Hi, I'm ${cfg.agent_name} from ${cfg.business_name}`;
            document.getElementById("callback-title").textContent = `Get a call from ${cfg.agent_name}`;
        }
        document.getElementById("callback-btn").hidden = !cfg.callback_enabled;
        callbackReady = Boolean(cfg.callback_ready);
        document.getElementById("kb-btn").hidden = !cfg.kb_uploads;
    } catch (_) { /* keep default */ }
}

async function fetchSystemInfo() {
    try {
        const res = await fetch('/health');
        if (!res.ok) return;
        const data = await res.json();
        const e = data.engines || {};

        // Header badge value spans (full model name)
        const hdrLlm = document.getElementById("hdr-llm");
        const hdrStt = document.getElementById("hdr-stt");
        const hdrTts = document.getElementById("hdr-tts");
        if (hdrLlm) hdrLlm.textContent = e.llm || '—';
        if (hdrStt) hdrStt.textContent = e.stt || '—';
        if (hdrTts) hdrTts.textContent = e.tts || '—';

        // Also set title attr so hovering shows full name when truncated
        const llmBadge = document.getElementById("ui-llm-model");
        const sttBadge = document.getElementById("ui-stt-model");
        const ttsBadge = document.getElementById("ui-tts-model");
        if (llmBadge) llmBadge.title = e.llm || '';
        if (sttBadge) sttBadge.title = e.stt || '';
        if (ttsBadge) ttsBadge.title = e.tts || '';

        // HUD cells (full name — panel is wide enough now)
        const hLlm = document.getElementById("hud-llm");
        const hStt = document.getElementById("hud-stt");
        const hTts = document.getElementById("hud-tts");
        if (hLlm) hLlm.textContent = e.llm || '—';
        if (hStt) hStt.textContent = e.stt || '—';
        if (hTts) hTts.textContent = e.tts || '—';

        // Mobile model strip
        const mLlm = document.getElementById("mob-llm");
        const mStt = document.getElementById("mob-stt");
        const mTts = document.getElementById("mob-tts");
        if (mLlm) mLlm.textContent = e.llm || '—';
        if (mStt) mStt.textContent = e.stt || '—';
        if (mTts) mTts.textContent = e.tts || '—';
    } catch (err) {
        console.warn("Health check failed:", err);
    }
}

// ==========================================================================
// WEBSOCKET
// ==========================================================================

function setupWebSocket() {
    updateConnectionStatus("connecting");
    ws = new WebSocket(socketUrl);

    ws.onopen = () => {
        updateConnectionStatus("connected");
        setUIState("idle");
        enableControls(true);
        _wsReconnectDelay = 1000;
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "status") {
            if (data.status === "thinking") {
                setUIState("thinking");
                isStreamFinished = false;
                audioQueue = [];
                isPlayingAudioQueue = false;
                prepareAssistantBubble();
                currentAssistantText = "";
            }

        } else if (data.type === "transcription") {
            appendUserMessage(data.text);

        } else if (data.type === "chunk") {
            setUIState("speaking");
            appendAssistantChunk(data.text);

        } else if (data.type === "audio_chunk") {
            if (isListeningEnabled && !scriptProcessor) startListening();
            setUIState("speaking");
            queueAudioChunk(data.audio, data.format || "wav", data.text);

        } else if (data.type === "done") {
            if (data.source) setAssistantSource(data.source);
            isStreamFinished = true;
            if (audioQueue.length === 0 && !isPlayingAudioQueue) {
                finishSpeakingCycle();
            }

        } else if (data.type === "booking") {
            appendBookingCard(data.booking);

        } else if (data.type === "error") {
            setUIState(isListeningEnabled ? "listening" : "idle");
            appendSystemMessage(data.text || "Unknown error");
            if (isListeningEnabled) setTimeout(startListening, 800);

        } else if (data.type === "interrupt") {
            // acknowledged

        } else if (data.type === "metrics") {
            console.debug("[VoiceAgent] turn latency (ms):", data.timings_ms);
        }
    };

    ws.onclose = (event) => {
        console.log(`WebSocket closed (code ${event.code}) — reconnecting in ${_wsReconnectDelay / 1000}s`);
        updateConnectionStatus("disconnected");
        enableControls(false);
        setUIState("idle");
        stopMediaRecorderAndContext();
        stopWaveform();

        if (event.code === 1008) {   // bad or missing ACCESS_TOKEN: retrying won't help
            appendSystemMessage("Access denied. Open this page with the link that includes ?token=…");
            return;
        }
        if (event.code === 1013 && _wsReconnectDelay === 1000) {
            appendSystemMessage("The assistant is busy with other customers. Reconnecting…");
        }

        setTimeout(() => {
            if (!ws || ws.readyState === WebSocket.CLOSED) setupWebSocket();
        }, _wsReconnectDelay);

        _wsReconnectDelay = Math.min(_wsReconnectDelay * 2, _WS_RECONNECT_MAX);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
}

// ==========================================================================
// AUDIO PLAYBACK QUEUE
// ==========================================================================

function playNextAudioChunk() {
    if (audioQueue.length === 0) {
        isPlayingAudioQueue = false;
        if (isStreamFinished) finishSpeakingCycle();
        return;
    }

    const chunk = audioQueue.shift();
    const b64 = typeof chunk === 'string' ? chunk : chunk.b64Audio;
    const fmt = typeof chunk === 'string' ? 'wav' : chunk.format;
    const mime = fmt === "mp3" ? "audio/mpeg" : "audio/wav";

    if (activeAudio) { try { activeAudio.pause(); } catch (_) { } }

    activeAudio = new Audio(`data:${mime};base64,${b64}`);
    activeAudio.onended = () => { activeAudio = null; playNextAudioChunk(); };
    activeAudio.onerror = () => { activeAudio = null; playNextAudioChunk(); };
    activeAudio.play().catch(() => { activeAudio = null; playNextAudioChunk(); });
}

function queueAudioChunk(b64Audio, format = "wav", text = null) {
    audioQueue.push({ b64Audio, format, text });
    if (!isPlayingAudioQueue) {
        isPlayingAudioQueue = true;
        playNextAudioChunk();
    }
}

// ==========================================================================
// AUDIO HELPERS
// ==========================================================================

function downsample(buffer, fromRate, toRate) {
    if (fromRate === toRate) return buffer;
    const ratio = fromRate / toRate;
    const length = Math.round(buffer.length / ratio);
    const result = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        const pos = i * ratio;
        const idx = Math.floor(pos);
        const frac = pos - idx;
        result[i] = (buffer[idx] || 0) + frac * ((buffer[idx + 1] || 0) - (buffer[idx] || 0));
    }
    return result;
}

function encodeWav16(samples, sampleRate) {
    const dataLen = samples.length * 2;
    const buf = new ArrayBuffer(44 + dataLen);
    const view = new DataView(buf);
    const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
    writeStr(0, 'RIFF'); view.setUint32(4, 36 + dataLen, true);
    writeStr(8, 'WAVE'); writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);       // PCM fmt chunk size
    view.setUint16(20, 1, true);       // PCM format
    view.setUint16(22, 1, true);       // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);       // block align (16-bit mono = 2 bytes)
    view.setUint16(34, 16, true);       // bits per sample
    writeStr(36, 'data'); view.setUint32(40, dataLen, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        off += 2;
    }
    return buf;
}

// ==========================================================================
// SPEECH RECOGNITION & MIC
// ==========================================================================

function setupSpeechRecognition() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        appendSystemMessage("Your browser does not support audio recording.");
    }
}

function stopMediaRecorderAndContext() {
    if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    if (scriptProcessor) {
        try { scriptProcessor.disconnect(); } catch (_) { }
        scriptProcessor = null;
    }
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
    }
    if (audioContext && audioContext.state !== "closed") {
        try { audioContext.close(); } catch (_) { }
    }
    audioContext = null;
    analyser = null;
    pcmBuffer = [];
}

async function startListening() {
    if (!isListeningEnabled) return;

    stopMediaRecorderAndContext();
    setUIState("listening");
    pcmBuffer = [];
    speechDetected = false;

    try {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });
    } catch (err) {
        console.error("Mic access error:", err);
        appendSystemMessage("Microphone access denied: " + err.message);
        stopSystem();
        return;
    }

    const TARGET_SR = 16000;
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    micSource = audioContext.createMediaStreamSource(micStream);
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

    // Route through a muted gain node so onaudioprocess fires without feedback
    const muteGain = audioContext.createGain();
    muteGain.gain.value = 0;
    micSource.connect(analyser);
    micSource.connect(scriptProcessor);
    scriptProcessor.connect(muteGain);
    muteGain.connect(audioContext.destination);

    const inputSR = audioContext.sampleRate;
    scriptProcessor.onaudioprocess = (e) => {
        if (!isListeningEnabled) return;
        const input = e.inputBuffer.getChannelData(0);
        const resampled = downsample(input, inputSR, TARGET_SR);
        for (let i = 0; i < resampled.length; i++) pcmBuffer.push(resampled[i]);
    };

    analyser.fftSize = 512;
    const bufLen = analyser.frequencyBinCount;
    const dataArr = new Uint8Array(bufLen);

    const SILENCE_MS = silenceMs;   // end-of-speech pause, from /config (VAD_SILENCE_MS)
    const SPEECH_START_FRAMES = 4;
    const BARGE_IN_FRAMES = 12;
    const BARGE_IN_MULTIPLIER = 2.5;
    const BARGE_IN_COOLDOWN = 1500;

    let loudFrames = 0;
    let calibFrames = 0;
    let noiseFloor = 0.02;
    bargedIn = false;

    startWaveform();

    function checkAudio() {
        if (!isListeningEnabled) return;

        analyser.getByteTimeDomainData(dataArr);
        let sq = 0;
        for (let i = 0; i < bufLen; i++) {
            const n = (dataArr[i] - 128) / 128;
            sq += n * n;
        }
        const rms = Math.sqrt(sq / bufLen);

        if (calibFrames < 30) { noiseFloor = Math.max(noiseFloor, rms * 1.5); calibFrames++; }
        const threshold = Math.max(noiseFloor, 0.04);

        if (rms > threshold) {
            loudFrames++;

            const sinceStarted = Date.now() - speakingStartTime;
            if (
                appState === "speaking"
                && !bargedIn
                && loudFrames >= BARGE_IN_FRAMES
                && rms > threshold * BARGE_IN_MULTIPLIER
                && sinceStarted > BARGE_IN_COOLDOWN
            ) {
                bargedIn = true;
                pcmBuffer = [];       // discard pre-barge audio (contains bot echo)
                speechDetected = true;     // user is actively speaking — skip the start threshold
                stopAllSpeech();
                setUIState("listening");   // give immediate UI feedback
                if (ws && ws.readyState === WebSocket.OPEN)
                    ws.send(JSON.stringify({ type: "interrupt" }));
            }

            if (!speechDetected && loudFrames >= SPEECH_START_FRAMES) speechDetected = true;
            if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
        } else {
            loudFrames = 0;
            if (speechDetected && !silenceTimer) {
                silenceTimer = setTimeout(() => {
                    silenceTimer = null;
                    if (pcmBuffer.length > 0 && isListeningEnabled) {
                        const samples = new Float32Array(pcmBuffer);
                        pcmBuffer = [];
                        speechDetected = false;
                        sendPcmToServer(samples, TARGET_SR);
                    }
                }, SILENCE_MS);
            }
        }
        animationFrameId = requestAnimationFrame(checkAudio);
    }
    animationFrameId = requestAnimationFrame(checkAudio);
}

// ==========================================================================
// CANVAS WAVEFORM
// ==========================================================================

function initWaveformCanvas() {
    waveCanvas = document.getElementById("waveform-canvas");
    if (!waveCanvas) return;
    waveCtx = waveCanvas.getContext("2d");
    resizeWaveCanvas();
    window.addEventListener("resize", resizeWaveCanvas);
}

function resizeWaveCanvas() {
    if (!waveCanvas) return;
    waveCanvas.width = waveCanvas.offsetWidth || 340;
    waveCanvas.height = waveCanvas.offsetHeight || 52;
}

function startWaveform() {
    if (!waveCanvas || !waveCtx) return;
    waveCanvas.classList.add("active");
    drawWaveform();
}

function stopWaveform() {
    if (waveAnimId) { cancelAnimationFrame(waveAnimId); waveAnimId = null; }
    if (waveCtx && waveCanvas)
        waveCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
    if (waveCanvas) waveCanvas.classList.remove("active");
}

function drawWaveform() {
    if (!waveCtx || !analyser) { stopWaveform(); return; }

    const bufLen = analyser.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyser.getByteTimeDomainData(data);

    const w = waveCanvas.width, h = waveCanvas.height;
    waveCtx.clearRect(0, 0, w, h);

    // Color follows the theme tokens in index.html
    const css = getComputedStyle(document.documentElement);
    const color = (appState === "listening" ? css.getPropertyValue("--listen")
        : appState === "speaking" ? css.getPropertyValue("--accent")
            : css.getPropertyValue("--text-dim")).trim();

    waveCtx.lineWidth = 1.5;
    waveCtx.strokeStyle = color;

    waveCtx.beginPath();
    const sliceW = w / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
        const v = data[i] / 128.0;
        const y = (v * h) / 2;
        i === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y);
        x += sliceW;
    }
    waveCtx.lineTo(w, h / 2);
    waveCtx.stroke();

    waveAnimId = requestAnimationFrame(drawWaveform);
}

// ==========================================================================
// FLOW
// ==========================================================================

function finishSpeakingCycle() {
    stopWaveform();
    if (isListeningEnabled) {
        setUIState("listening");
        startListening();
    } else {
        setUIState("idle");
    }
}

function stopAllSpeech() {
    if (activeAudio) { try { activeAudio.pause(); } catch (_) { } activeAudio = null; }
    isSpeaking = false;
    audioQueue = [];
    isPlayingAudioQueue = false;
    stopWaveform();
}

async function sendTextToServer(text) {
    if (!text.trim()) return;
    unlockAudio();
    stopMediaRecorderAndContext();
    stopAllSpeech();
    isStreamFinished = false;

    const wm = document.getElementById("welcome-msg");
    if (wm) wm.remove();
    appendUserMessage(text);
    setUIState("thinking");

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "text", text, voice: selectedVoiceName }));
    } else {
        setUIState("idle");
        appendSystemMessage("Not connected. Please wait for reconnection.");
    }
}

function sendPcmToServer(samples, sampleRate) {
    stopMediaRecorderAndContext();
    stopAllSpeech();
    isStreamFinished = false;
    setUIState("thinking");

    if (!ws || ws.readyState !== WebSocket.OPEN) {
        setUIState("idle");
        appendSystemMessage("Not connected. Please wait for reconnection.");
        return;
    }
    ws.send(encodeWav16(samples, sampleRate));
}

function startSystem() {
    isListeningEnabled = true;
    stopAllSpeech();
    startListening();
    toggleMicBtn.classList.add("active");
    document.getElementById("mic-icon").className = "fa-solid fa-microphone-slash";
    document.getElementById("mic-btn-text").textContent = "Stop listening";
    const fab = document.getElementById("mobile-mic-fab");
    if (fab) {
        fab.classList.add("active");
        const fabIcon = document.getElementById("fab-icon");
        const fabLabel = document.getElementById("fab-label");
        if (fabIcon) fabIcon.className = "fa-solid fa-microphone-slash";
        if (fabLabel) fabLabel.textContent = "Stop";
    }
}

function stopSystem() {
    isListeningEnabled = false;
    stopMediaRecorderAndContext();
    stopAllSpeech();
    setUIState("idle");
    toggleMicBtn.classList.remove("active");
    document.getElementById("mic-icon").className = "fa-solid fa-microphone";
    document.getElementById("mic-btn-text").textContent = "Start listening";
    const fab = document.getElementById("mobile-mic-fab");
    if (fab) {
        fab.classList.remove("active");
        const fabIcon = document.getElementById("fab-icon");
        const fabLabel = document.getElementById("fab-label");
        if (fabIcon) fabIcon.className = "fa-solid fa-microphone";
        if (fabLabel) fabLabel.textContent = "Speak";
    }
}

// ==========================================================================
// EVENT LISTENERS
// ==========================================================================

function setupEventListeners() {
    toggleMicBtn.onclick = () => isListeningEnabled ? stopSystem() : startSystem();
    orb.onclick = () => isListeningEnabled ? stopSystem() : startSystem();

    // Mobile floating mic button
    const mobileFab = document.getElementById("mobile-mic-fab");
    if (mobileFab) mobileFab.onclick = () => isListeningEnabled ? stopSystem() : startSystem();

    clearBtn.onclick = () => {
        stopAllSpeech();
        chatContainer.innerHTML = `
            <div id="welcome-msg" class="welcome-card">
                <h3>Conversation cleared</h3>
                <p>Tap the microphone or type below to start again.</p>
            </div>`;
    };

    voiceSelect.onchange = (e) => {
        selectedVoiceName = e.target.value;
        try { localStorage.setItem("voiceagent-voice", selectedVoiceName); } catch (_) { }
    };

    if (chatForm && chatInput) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const text = chatInput.value.trim();
            if (text) { sendTextToServer(text); chatInput.value = ""; }
        });
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                const text = chatInput.value.trim();
                if (text) { sendTextToServer(text); chatInput.value = ""; }
            }
        });
    }
}

function enableControls(enabled) {
    if (toggleMicBtn) toggleMicBtn.disabled = !enabled;
    if (orb) orb.disabled = !enabled;
    if (chatInput) chatInput.disabled = !enabled;
    const sub = document.getElementById("chat-submit");
    if (sub) sub.disabled = !enabled;
    const fab = document.getElementById("mobile-mic-fab");
    if (fab) fab.disabled = !enabled;
}

// ==========================================================================
// UI STATE
// ==========================================================================

function updateConnectionStatus(status) {
    connectionStatus.className = status;
    const text = { connected: "Connected", connecting: "Connecting…", disconnected: "Disconnected" };
    document.getElementById("conn-text").textContent = text[status] || status;
}

function setUIState(state) {
    appState = state;

    // Orb class
    orb.className = state;

    // Body class for ring color cascading
    document.body.className = "state-" + state;

    // Remove welcome on first activity
    if (state !== "idle") {
        const wm = document.getElementById("welcome-msg");
        if (wm) wm.remove();
    }

    switch (state) {
        case "idle":
            stateTag.className = "";
            stateTag.textContent = "Ready";
            statePrompt.textContent = "Tap the microphone to start";
            orbIcon.innerHTML = '<i class="fa-solid fa-microphone"></i>';
            stopWaveform();
            break;

        case "listening":
            stateTag.className = "listening";
            stateTag.textContent = "Listening";
            statePrompt.textContent = "Go ahead, I'm listening…";
            orbIcon.innerHTML = '<i class="fa-solid fa-microphone-lines"></i>';
            break;

        case "thinking":
            stateTag.className = "thinking";
            stateTag.textContent = "Thinking";
            statePrompt.textContent = "Working on it…";
            orbIcon.innerHTML = '<i class="fa-solid fa-circle-notch" style="animation:spin 1s linear infinite"></i>';
            stopWaveform();
            break;

        case "speaking":
            speakingStartTime = Date.now();
            bargedIn = false;
            stateTag.className = "speaking";
            stateTag.textContent = "Speaking";
            statePrompt.textContent = "Talk over me to interrupt";
            orbIcon.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            if (analyser) startWaveform();
            break;
    }
}

// ==========================================================================
// CHAT RENDERING
// ==========================================================================

function appendUserMessage(text) {
    const el = document.createElement("div");
    el.className = "user-msg";
    el.innerHTML = `
        <div class="user-bubble">
            <div class="msg-label">You</div>
            <div class="user-text">${escapeHTML(text)}</div>
        </div>`;
    chatContainer.appendChild(el);
    scrollToBottom();
}

function prepareAssistantBubble() {
    currentAssistantText = "";
    currentAssistantBubble = document.createElement("div");
    currentAssistantBubble.className = "agent-msg";
    currentAssistantBubble.innerHTML = `
        <div class="agent-header">
            <span class="msg-label">${escapeHTML(businessLabel)}</span>
            <span id="latest-assistant-source" class="src-badge" style="display:none"></span>
        </div>
        <div id="latest-assistant-response" class="agent-text">
            <div class="thinking-dots">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    chatContainer.appendChild(currentAssistantBubble);
    scrollToBottom();
}

function appendAssistantChunk(chunk) {
    if (!currentAssistantBubble) return;
    currentAssistantText += chunk;
    const box = currentAssistantBubble.querySelector("#latest-assistant-response");
    if (box) box.innerHTML = escapeHTML(currentAssistantText);
    scrollToBottom();
}

function setAssistantSource(source) {
    if (!currentAssistantBubble) return;
    const badge = currentAssistantBubble.querySelector("#latest-assistant-source");
    if (!badge) return;

    const labels = {
        web: "Web search",
        kb: "Knowledge base",
        none: "Model knowledge",
    };
    badge.textContent = labels[source] || source;
    badge.className = `src-badge ${source}`;
    badge.style.display = "inline";
}

function appendBookingCard(b) {
    if (!b) return;
    const kind = b.kind === "meeting" ? "Sales meeting" : "Test drive";
    const rows = [
        ["Car", b.car_model], ["Showroom", b.showroom],
        ["When", `${b.weekday}, ${b.date} at ${b.time}`], ["Name", b.customer_name],
        ["Mobile", b.phone_last4 ? `•••••• ${b.phone_last4}` : ""],
    ].filter(([, v]) => v);
    const el = document.createElement("div");
    el.className = "booking-card";
    el.innerHTML = `
        <div class="booking-head">
            <span><i class="fa-regular fa-calendar-check"></i> ${kind} confirmed</span>
            <span class="booking-id">${escapeHTML(b.booking_id || "")}</span>
        </div>
        <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHTML(String(v))}</dd>`).join("")}</dl>`;
    chatContainer.appendChild(el);
    scrollToBottom();
}

function appendSystemMessage(text) {
    const el = document.createElement("div");
    el.className = "system-msg";
    el.innerHTML = `<div class="system-bubble"><i class="fa-solid fa-triangle-exclamation"></i> ${escapeHTML(text)}</div>`;
    chatContainer.appendChild(el);
    scrollToBottom();
}

// ==========================================================================
// DIALOGS — call me back, knowledge base documents
// ==========================================================================

function setupDialogClose(dialog) {
    dialog.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => dialog.close()));
    dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });   // backdrop
}

function setStatus(el, text, kind = "") {
    el.textContent = text;
    el.className = `modal-status ${kind}`;
}

async function readError(res) {
    try { return (await res.json()).detail || `Error ${res.status}`; }
    catch (_) { return `Error ${res.status}`; }
}

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
}

// ── Call me back: the AI agent phones the customer ─────────────────────────

let callbackReady = false;   // /config: Exotel outbound calling is configured

function setupCallback() {
    const dialog = document.getElementById("callback-dialog");
    const form = document.getElementById("callback-form");
    const phone = document.getElementById("callback-phone");
    const status = document.getElementById("callback-status");
    const submit = document.getElementById("callback-submit");
    setupDialogClose(dialog);

    document.getElementById("callback-btn").addEventListener("click", () => {
        setStatus(status, "");
        submit.disabled = false;
        dialog.showModal();
        phone.focus();
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!phone.value.trim()) {
            setStatus(status, "Please enter your mobile number.", "err");
            return;
        }
        if (!callbackReady) {
            setStatus(status, "Phone call-backs aren't set up on this server yet. " +
                              "Please talk or type here instead.", "err");
            return;
        }
        submit.disabled = true;
        setStatus(status, "Requesting your call…");
        try {
            const res = await fetch(withPageToken("/callback"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ phone: phone.value }),
            });
            if (!res.ok) {
                setStatus(status, await readError(res), "err");
                submit.disabled = false;
                return;
            }
            const last4 = phone.value.replace(/\D/g, "").slice(-4);
            setStatus(status, `Calling ••••••${last4} now. Please keep your phone nearby.`, "ok");
        } catch (_) {
            setStatus(status, "Network error. Please try again.", "err");
            submit.disabled = false;
        }
    });
}

// ── Knowledge base documents (upload / delete) ─────────────────────────────

// Adds the page's ACCESS_TOKEN (if any) to an API path.
function withPageToken(path) {
    return accessToken ? `${path}?token=${encodeURIComponent(accessToken)}` : path;
}

function setupKnowledgeBase() {
    const dialog = document.getElementById("kb-dialog");
    const drop = document.getElementById("kb-drop");
    const fileInput = document.getElementById("kb-file");
    const list = document.getElementById("kb-list");
    const status = document.getElementById("kb-status");
    const limits = document.getElementById("kb-limits");
    setupDialogClose(dialog);

    async function loadDocs() {
        let res;
        try { res = await fetch(withPageToken("/kb/documents")); }
        catch (_) { setStatus(status, "Network error.", "err"); return false; }
        if (!res.ok) { setStatus(status, await readError(res), "err"); return false; }
        const data = await res.json();
        drop.hidden = !data.uploads_enabled;
        limits.textContent = `.md, .txt, .pdf · up to ${data.max_mb} MB each`;
        if (!data.uploads_enabled) {
            setStatus(status, "This server uses Google Discovery Engine, so uploads are turned off here.", "err");
        }
        renderDocs(data.documents);
        return true;
    }

    function renderDocs(docs) {
        list.innerHTML = "";
        if (!docs.length) {
            list.innerHTML = `<li class="kb-empty">No documents yet.</li>`;
            return;
        }
        for (const d of docs) {
            const li = document.createElement("li");
            li.innerHTML = `<i class="fa-regular fa-file-lines"></i>
                <span class="kb-name">${escapeHTML(d.name)}</span>
                <span class="kb-meta">${formatSize(d.size)}</span>
                <span class="kb-tag ${d.uploaded ? "uploaded" : ""}">${d.uploaded ? "Uploaded" : "Built-in"}</span>`;
            if (d.uploaded) {
                const del = document.createElement("button");
                del.type = "button";
                del.className = "kb-del";
                del.title = `Delete ${d.name}`;
                del.setAttribute("aria-label", del.title);
                del.innerHTML = `<i class="fa-regular fa-trash-can"></i>`;
                del.addEventListener("click", () => deleteDoc(d.name));
                li.appendChild(del);
            }
            list.appendChild(li);
        }
    }

    async function uploadFiles(files) {
        files = [...files];
        if (!files.length) return;
        const errors = [];
        let added = 0, chunks = 0;
        for (const [i, file] of files.entries()) {
            setStatus(status, `Uploading ${file.name} (${i + 1} of ${files.length})…`);
            try {
                const res = await fetch(withPageToken(`/kb/documents/${encodeURIComponent(file.name)}`),
                                        { method: "PUT", body: file });
                if (!res.ok) { errors.push(`${file.name}: ${await readError(res)}`); continue; }
                chunks = (await res.json()).chunks;
                added++;
            } catch (_) {
                errors.push(`${file.name}: network error`);
            }
        }
        fileInput.value = "";
        if (!(await loadDocs())) return;
        if (errors.length) setStatus(status, errors.join(" · "), "err");
        else setStatus(status, `Added ${added} document${added === 1 ? "" : "s"}. ` +
                               `The assistant uses them now (${chunks} sections indexed).`, "ok");
    }

    async function deleteDoc(name) {
        if (!window.confirm(`Delete ${name} from the knowledge base?`)) return;
        try {
            const res = await fetch(withPageToken(`/kb/documents/${encodeURIComponent(name)}`),
                                    { method: "DELETE" });
            if (!res.ok) { setStatus(status, await readError(res), "err"); return; }
            if (await loadDocs()) setStatus(status, `Deleted ${name}.`, "ok");
        } catch (_) {
            setStatus(status, "Network error.", "err");
        }
    }

    document.getElementById("kb-btn").addEventListener("click", () => {
        setStatus(status, "");
        dialog.showModal();
        loadDocs();
    });
    fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
    drop.addEventListener("drop", (e) => {
        e.preventDefault();
        drop.classList.remove("dragover");
        uploadFiles(e.dataTransfer.files);
    });
}

// ── helpers ────────────────────────────────────────────────────────────────

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHTML(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ==========================================================================
// BOOTSTRAP
// ==========================================================================
window.onload = init;
