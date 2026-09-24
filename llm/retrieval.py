"""
llm/retrieval.py
Context retrieval with latency-aware, multilingual routing.

Plan per query (RETRIEVAL_MODE=auto, the default):
  1. Short conversational turns ("hi", "thanks", "ठीक है" …) → no retrieval.
  2. Real-time / live-data queries in any supported language → web search.
  3. Everything else → knowledge base (if GCP Discovery Engine is configured,
     raced against web search), otherwise no retrieval: the LLM answers from
     its own knowledge immediately instead of waiting on a search it rarely needs.

RETRIEVAL_MODE=always restores "retrieve for every substantive query";
RETRIEVAL_MODE=off disables retrieval.

The orchestrator waits at most LLM_RETRIEVAL_WAIT seconds; results are cached
for RETRIEVAL_CACHE_TTL seconds and concurrent identical queries share one fetch.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from core.lang import INDIC_SCRIPT_RE as _INDIC_SCRIPT_RE
from llm.web_search import WebSearchService

logger = get_logger("llm.retrieval")

_NONE = {"source": "none", "context": ""}
_CACHE_MAX = 2000
_MAX_CONTEXT_CHARS = 3000

# ── Short conversational openers — skip retrieval ─────────────────────────────
# Note: startswith check + word-count guard in should_retrieve() means that
# "Namaste, aaj ka gold rate?" (starts with a greeting but is long) still
# goes to retrieval.
_CONVERSATIONAL_PREFIXES = (
    # English
    "hi", "hello", "hey", "thanks", "thank you",
    "good morning", "good afternoon", "good evening",
    "how are you", "bye", "goodbye", "ok", "okay",
    "sure", "great", "alright", "nice", "wow",
    # Hindi – Romanized
    "haan", "nahi", "nahin", "shukriya", "dhanyawad", "dhanyavaad",
    "namaste", "namaskar", "jai hind", "jai ho",
    "theek hai", "bilkul", "acha", "accha", "wah",
    # Hindi – Devanagari
    "हाँ", "हां", "नहीं", "ठीक है", "धन्यवाद", "शुक्रिया",
    "नमस्ते", "नमस्कार", "वाह",
    # Tamil
    "vanakkam", "nandri", "sari",
    # Telugu
    "namaskaram", "dhanyavaadalu", "sare",
    # Kannada
    "namaskara", "dhanyavada",
    # Bengali
    "dhonyobad", "namaskar",
)

# ── Queries that need live data → web search
# Very generic words ("now", "अब", "नया") are deliberately excluded: they made
# ordinary questions wait on a web search. ────────────────────────────────────
# Covers English, Romanized Hindi, Devanagari, Tamil, Telugu, Kannada,
# Bengali, Gujarati, Marathi, Punjabi.
_REALTIME_RE = re.compile(
    r"("
    # ── English ────────────────────────────────────────────────────────────────
    r"\b(latest|current|today|tonight|right now|just now|live|breaking|"
    r"this week|this month|this year|yesterday|tomorrow|recent|"
    r"news|headline|update|score|result|election|"
    r"price|rate|cost|stock|share\s*price|market|crypto|bitcoin|ethereum|nft|"
    r"weather|forecast|temperature|rain|humidity|wind|flood|cyclone|"
    r"traffic|flight|train|bus|petrol|diesel|fuel|lpg|gas)\b"
    r"|"
    # ── Romanized Hindi (typed in Latin) ──────────────────────────────────────
    r"\b(aaj|abhi|kal |parso|taaza|taza|"
    r"mausam|baarish|barish|badal|tapman|"
    r"bhav|daam|dam |kimat|keemat|sona|chandi|"
    r"petrol|diesel|lpg|"
    r"sensex|nifty|bazaar|bazar|share|"
    r"khabar|samachar|taza khabar|breaking|"
    r"score|result|jeet|haar|chunav|sarkar)\b"
    r"|"
    # ── Hindi – Devanagari ────────────────────────────────────────────────────
    r"(आज|कल|परसों|अभी|इस वक्त|इस समय|फिलहाल|"
    r"ताज़ा|ताजा|"
    r"मौसम|बारिश|तापमान|बाढ़|तूफान|ठंड|ठंडा|गर्मी|"
    r"भाव|दाम|कीमत|रेट|"
    r"सोना|सोने|चांदी|चाँदी|"
    r"पेट्रोल|डीजल|गैस|एलपीजी|"
    r"शेयर|सेंसेक्स|निफ्टी|बाजार|बाज़ार|क्रिप्टो|बिटकॉइन|"
    r"खबर|समाचार|"
    r"स्कोर|रिजल्ट|मैच|जीत|हार|"
    r"चुनाव|सरकार|नेता|"
    r"ट्रैफिक|फ्लाइट|ट्रेन)"
    r"|"
    # ── Tamil ────────────────────────────────────────────────────────────────
    r"(இன்று|இப்போது|இப்போ|நேற்று|நாளை|தற்போது|"
    r"வானிலை|மழை|வெப்பம்|வெள்ளம்|புயல்|"
    r"விலை|தங்கம்|வெள்ளி|பெட்ரோல்|டீசல்|"
    r"செய்தி|நேரடி|புதிய|சந்தை|பங்கு)"
    r"|"
    # ── Telugu ───────────────────────────────────────────────────────────────
    r"(ఈరోజు|ఇప్పుడు|నిన్న|రేపు|ప్రస్తుతం|"
    r"వాతావరణం|వర్షం|ఉష్ణోగ్రత|వరద|తుఫాను|"
    r"ధర|బంగారం|వెండి|పెట్రోల్|డీజల్|"
    r"వార్త|వార్తలు|తాజా|మార్కెట్|స్టాక్)"
    r"|"
    # ── Kannada ──────────────────────────────────────────────────────────────
    r"(ಇಂದು|ಈಗ|ನಿನ್ನೆ|ನಾಳೆ|ಪ್ರಸ್ತುತ|"
    r"ಹವಾಮಾನ|ಮಳೆ|ತಾಪಮಾನ|ಪ್ರವಾಹ|"
    r"ಬೆಲೆ|ಚಿನ್ನ|ಬೆಳ್ಳಿ|ಪೆಟ್ರೋಲ್|ಡೀಸೆಲ್|"
    r"ಸುದ್ದಿ|ಹೊಸ|ಮಾರುಕಟ್ಟೆ|ಷೇರು)"
    r"|"
    # ── Bengali ──────────────────────────────────────────────────────────────
    r"(আজ|এখন|গতকাল|আগামীকাল|বর্তমান|"
    r"আবহাওয়া|বৃষ্টি|তাপমাত্রা|বন্যা|"
    r"দাম|সোনা|রুপা|পেট্রোল|ডিজেল|"
    r"খবর|সংবাদ|নতুন|বাজার|শেয়ার)"
    r"|"
    # ── Gujarati ─────────────────────────────────────────────────────────────
    r"(આજ|હવે|ગઈકાલ|આવતીકાલ|અત્યારે|"
    r"હવામાન|વરસાદ|તાપમાન|"
    r"ભાવ|સોનુ|ચાંદી|પેટ્રોલ|ડીઝલ|"
    r"સમાચાર|ખબર|નવી|બજાર)"
    r"|"
    # ── Marathi ──────────────────────────────────────────────────────────────
    r"(आता|काल|उद्या|सध्या|"
    r"हवामान|पाऊस|तापमान|"
    r"किंमत|सोने|चांदी|पेट्रोल|डिझेल|"
    r"बातमी|बातम्या|नवीन|बाजार)"
    r"|"
    # ── Punjabi / Gurmukhi ────────────────────────────────────────────────────
    r"(ਅੱਜ|ਹੁਣ|ਕੱਲ੍ਹ|ਮੌਸਮ|ਮੀਂਹ|ਕੀਮਤ|ਸੋਨਾ|ਖ਼ਬਰ|ਬਾਜ਼ਾਰ)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# ── Keyword map: Indic → English for fallback search ─────────────────────────
_INDIC_TO_EN: dict[str, str] = {
    # Devanagari
    "आज": "today", "कल": "yesterday", "अभी": "now", "अब": "now",
    "मौसम": "weather", "बारिश": "rain", "तापमान": "temperature",
    "भाव": "price", "दाम": "price", "कीमत": "price",
    "सोना": "gold", "सोने": "gold", "चांदी": "silver", "चाँदी": "silver",
    "पेट्रोल": "petrol", "डीजल": "diesel",
    "खबर": "news", "समाचार": "news", "ताज़ा": "latest", "ताजा": "latest",
    "शेयर": "shares", "बाजार": "market", "क्रिप्टो": "crypto",
    "बिटकॉइन": "bitcoin", "स्कोर": "score", "रिजल्ट": "result",
    "चुनाव": "election", "सरकार": "government",
    # Tamil
    "இன்று": "today", "விலை": "price", "வானிலை": "weather",
    "செய்தி": "news", "தங்கம்": "gold", "வெள்ளி": "silver",
    # Telugu
    "ఈరోజు": "today", "ధర": "price", "వాతావరణం": "weather",
    "వార్త": "news", "బంగారం": "gold", "వెండి": "silver",
    # Kannada
    "ಇಂದು": "today", "ಬೆಲೆ": "price", "ಹವಾಮಾನ": "weather",
    "ಸುದ್ದಿ": "news", "ಚಿನ್ನ": "gold", "ಬೆಳ್ಳಿ": "silver",
    # Bengali
    "আজ": "today", "দাম": "price", "আবহাওয়া": "weather",
    "খবর": "news", "সোনা": "gold", "রুপা": "silver",
    # Gujarati
    "આજ": "today", "ભાવ": "price", "હવામાન": "weather",
    "સમાચાર": "news", "સોનુ": "gold", "ચાંદી": "silver",
    # Marathi
    "आता": "now", "किंमत": "price", "हवामान": "weather",
    "बातमी": "news", "सोने": "gold",
    # Punjabi
    "ਅੱਜ": "today", "ਮੌਸਮ": "weather", "ਕੀਮਤ": "price",
    "ਸੋਨਾ": "gold", "ਖ਼ਬਰ": "news",
}


def _indic_to_english_query(query: str) -> str:
    """English keyword query from an Indic-script query ('' if nothing useful)."""
    parts: list[str] = []
    for indic, english in _INDIC_TO_EN.items():
        if indic in query and english not in parts:
            parts.append(english)
    latin = re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7F]", " ", query)).strip()
    if latin:
        parts.insert(0, latin)
    return " ".join(parts).strip()


def is_conversational(query: str) -> bool:
    q = query.strip().lower()
    if "?" in q or len(q.split()) > 6:
        return False
    return any(q.startswith(p) for p in _CONVERSATIONAL_PREFIXES)


def is_realtime(query: str) -> bool:
    return bool(_REALTIME_RE.search(query))


class RetrievalService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.web = WebSearchService(self.s) if self.s.web_search_enabled else None
        self._kb_client = None
        self._serving_config = (
            f"projects/{self.s.gcp_project_id}/locations/{self.s.gcp_location}"
            f"/collections/default_collection/dataStores/{self.s.gcp_data_store_id}"
            f"/servingConfigs/default_search"
        )
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task] = {}
        logger.info("Retrieval | mode=%s kb=%s web=%s wait=%.1fs",
                    self.s.retrieval_mode,
                    "on" if self.s.kb_configured else "off",
                    "on" if self.web else "off",
                    self.s.retrieval_wait)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def warmup(self) -> None:
        if self.s.kb_configured:
            try:
                await asyncio.to_thread(self._get_kb_client)
            except Exception as exc:
                logger.error("Knowledge base unavailable (pip install -r requirements-kb.txt "
                             "and check GCP credentials): %r", exc)
        if self.web:
            await self.web.warmup()

    def _get_kb_client(self):
        if self._kb_client is None:
            from google.cloud import discoveryengine_v1  # type: ignore
            self._kb_client = discoveryengine_v1.SearchServiceClient()
        return self._kb_client

    # ── routing ───────────────────────────────────────────────────────────────

    def plan(self, query: str) -> tuple[str, ...]:
        """Which sources to query: () | ("web",) | ("kb",) | ("kb", "web")."""
        mode = self.s.retrieval_mode
        if mode == "off" or not query.strip() or is_conversational(query):
            return ()
        kb_ok = self.s.kb_configured and (self.s.kb_multilingual or not _INDIC_SCRIPT_RE.search(query))
        if is_realtime(query):
            return ("web",) if self.web else (("kb",) if kb_ok else ())
        sources: list[str] = []
        if kb_ok:
            sources.append("kb")
        if self.web and (mode == "always" or (kb_ok and mode == "auto")):
            sources.append("web")
        return tuple(sources)

    # ── public ────────────────────────────────────────────────────────────────

    async def get_context(self, query: str, wait: float | None = None) -> dict:
        """Best context available within `wait` seconds; never raises."""
        sources = self.plan(query)
        if not sources:
            return _NONE
        key = query.strip().lower()
        hit = self._cache.get(key)
        if hit and hit[0] > time.monotonic():
            self._cache.move_to_end(key)
            logger.info("⚡ retrieval cache hit")
            return hit[1]

        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._retrieve(query, sources))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        try:
            # shield: a slow fetch keeps running and fills the cache for next time.
            return await asyncio.wait_for(asyncio.shield(task), timeout=wait)
        except asyncio.TimeoutError:
            logger.info("Retrieval not ready after %.1fs (%s) — answering without it",
                        wait, "+".join(sources))
            return _NONE

    async def _retrieve(self, query: str, sources: tuple[str, ...]) -> dict:
        t0 = time.perf_counter()
        runners = {"kb": self._kb_search, "web": self._web_search}
        tasks = {asyncio.create_task(runners[s](query)): s for s in sources}
        result = _NONE
        try:
            for fut in asyncio.as_completed(tasks, timeout=self.s.retrieval_timeout):
                try:
                    source, ctx = await fut
                except asyncio.TimeoutError:
                    logger.warning("Retrieval hard timeout (%.1fs)", self.s.retrieval_timeout)
                    break
                if ctx and self._is_quality(ctx, query):
                    result = {"source": source, "context": ctx[:_MAX_CONTEXT_CHARS]}
                    break
        finally:
            for t in tasks:
                t.cancel()
        logger.info("Retrieval %s → %s (%d chars, %.0f ms)", "+".join(sources),
                    result["source"], len(result["context"]), (time.perf_counter() - t0) * 1000)
        self._cache_put(query.strip().lower(), result)
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_quality(ctx: str, query: str) -> bool:
        text = ctx.strip()
        if len(text) < 100:
            return False
        # English-only KB text for an Indic query is accepted only if substantial.
        if _INDIC_SCRIPT_RE.search(query) and not _INDIC_SCRIPT_RE.search(text):
            return len(text) >= 400
        return True

    def _cache_put(self, key: str, value: dict) -> None:
        ttl = self.s.retrieval_cache_ttl if value["source"] != "none" else 30
        self._cache[key] = (time.monotonic() + ttl, value)
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)

    async def _web_search(self, query: str) -> tuple[str, str]:
        ctx = await self.web.search(query)
        if not ctx and _INDIC_SCRIPT_RE.search(query):
            en_query = _indic_to_english_query(query)
            if en_query:
                logger.info("Indic web fallback — English keywords: %r", en_query[:60])
                ctx = await self.web.search(en_query)
        return "web", ctx

    async def _kb_search(self, query: str) -> tuple[str, str]:
        try:
            return "kb", await asyncio.to_thread(self._sync_kb_search, query)
        except Exception as exc:
            logger.error("KB search error: %s", exc)
            return "kb", ""

    def _sync_kb_search(self, query: str) -> str:
        from google.cloud import discoveryengine_v1  # type: ignore
        from google.protobuf.json_format import MessageToDict  # type: ignore

        request = discoveryengine_v1.SearchRequest(
            serving_config=self._serving_config, query=query, page_size=5)
        response = self._get_kb_client().search(request=request,
                                                timeout=self.s.retrieval_timeout)
        docs: list[str] = []
        for result in response.results:
            data = MessageToDict(result.document._pb)
            derived = data.get("derivedStructData", {})
            answers = derived.get("extractive_answers") or derived.get("extractiveAnswers", [])
            for answer in answers:
                content = answer.get("content", "")
                if content:
                    docs.append(f"Source: {derived.get('title', '')}\n{content}")
        return "\n\n".join(docs)
