"""
llm/retrieval.py
Context retrieval for the dealership agent: knowledge base first, Google second.

Plan per query (RETRIEVAL_MODE=auto, the default):
  1. Small talk ("hi", "thanks", "ठीक है" …) and turns containing a phone
     number or email → no retrieval (contact details never leave the server).
  2. Off-topic or unclear messages (llm/topic_guard.py) → knowledge base only;
     they never reach a search engine.
  3. Time-sensitive car questions that don't mention the business ("petrol
     price today in Pune") → web search only.
  4. Everything else → knowledge base; if it has no answer → web search.
     While a booking is being collected, web search is skipped so answers
     like a name or a date are never sent to a search engine.

Knowledge base: the local Markdown catalog in KB_DIR (BM25, < 1 ms) or Google
Discovery Engine when GCP_PROJECT_ID / GCP_DATA_STORE_ID are set. A slow
Discovery Engine lookup is hedged by starting the web search after 0.6 s.
Web search: Google via Gemini grounding, then Brave / scrapers (llm/web_search.py).

RETRIEVAL_MODE=always also tries the web for questions with no KB configured;
RETRIEVAL_MODE=off disables retrieval. Results are cached for
RETRIEVAL_CACHE_TTL seconds and concurrent identical queries share one fetch.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from core.lang import INDIC_SCRIPT_RE as _INDIC_SCRIPT_RE
from core.privacy import contains_contact_details
from llm.knowledge_base import LocalKnowledgeBase, tokenize
from llm.web_search import WebSearchService

logger = get_logger("llm.retrieval")

_NONE = {"source": "none", "context": ""}
_CACHE_MAX = 2000
_MAX_CONTEXT_CHARS = 3000
_WEB_HEDGE_S = 0.6       # start web search if a remote KB is still busy after this

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


# Words that tie a time-sensitive question to the dealership ("current offers on
# the Ion", "on-road price") so it still goes to the knowledge base first.
_BUSINESS_TERMS = frozenset(tokenize(
    "test drive showroom dealer dealership booking offer offers discount emi loan finance "
    "exchange warranty service servicing variant variants on-road ex-showroom delivery "
    "insurance accessories waiting period"))


class RetrievalService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.web = WebSearchService(self.s) if self.s.web_search_enabled else None
        self.kb_backend = self.s.kb_backend()
        self.local_kb: LocalKnowledgeBase | None = None
        if self.kb_backend == "local":
            self.local_kb = LocalKnowledgeBase(self.s.kb_dir, self.s.kb_min_coverage)
            if not self.local_kb.available:
                logger.warning("Local KB is empty (%s) — knowledge base disabled", self.s.kb_dir)
                self.kb_backend = "off"
        elif self.kb_backend == "discovery" and not self.s.discovery_configured:
            logger.warning("KB_PROVIDER=discovery but GCP_PROJECT_ID / GCP_DATA_STORE_ID "
                           "are not set — knowledge base disabled")
            self.kb_backend = "off"
        self._business_terms = _BUSINESS_TERMS | set(tokenize(" ".join(
            [self.s.business_name, *self.s.car_models, *self.s.showrooms])))
        self._kb_client = None
        self._serving_config = (
            f"projects/{self.s.gcp_project_id}/locations/{self.s.gcp_location}"
            f"/collections/default_collection/dataStores/{self.s.gcp_data_store_id}"
            f"/servingConfigs/default_search"
        )
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task] = {}
        logger.info("Retrieval | mode=%s kb=%s web=%s wait=%.1fs",
                    self.s.retrieval_mode, self.kb_backend,
                    self.web.describe() if self.web else "off", self.s.retrieval_wait)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @property
    def uses_local_kb(self) -> bool:
        """True when KB_PROVIDER resolves to the local Markdown KB (uploads apply)."""
        return self.s.kb_backend() == "local"

    def reload_local_kb(self) -> int:
        """Rebuild the local KB index after documents changed. Returns the chunk count."""
        kb = LocalKnowledgeBase(self.s.kb_dir, self.s.kb_min_coverage)
        self.local_kb = kb   # single assignment: in-flight searches keep the old index
        self.kb_backend = "local" if kb.available else "off"
        self._cache.clear()  # cached answers may be stale
        return len(kb.chunks)

    async def warmup(self) -> None:
        if self.kb_backend == "discovery":
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

    def mentions_business(self, query: str) -> bool:
        return bool(self._business_terms & set(tokenize(query)))

    def plan(self, query: str, booking_active: bool = False,
             allow_web: bool = True) -> tuple[str, ...]:
        """Sources in the order they are tried: () | ("kb",) | ("web",) | ("kb", "web").
        allow_web=False (off-topic or unclear messages, see llm/topic_guard.py)
        limits the lookup to the knowledge base."""
        mode = self.s.retrieval_mode
        if (mode == "off" or not query.strip() or is_conversational(query)
                or contains_contact_details(query)):
            return ()
        web_ok = self.web is not None and allow_web and not booking_active
        kb_ok = self.kb_backend == "local" or (
            self.kb_backend == "discovery"
            and (self.s.kb_multilingual or not _INDIC_SCRIPT_RE.search(query)))
        if web_ok and is_realtime(query) and not self.mentions_business(query):
            return ("web",)
        if kb_ok:
            return ("kb", "web") if web_ok else ("kb",)
        if web_ok and (is_realtime(query) or mode == "always"):
            return ("web",)
        return ()

    # ── public ────────────────────────────────────────────────────────────────

    async def get_context(self, query: str, wait: float | None = None,
                          sources: tuple[str, ...] | None = None) -> dict:
        """Best context available within `wait` seconds for `query` (the text
        that is searched; may include the previous question for follow-ups);
        never raises."""
        sources = self.plan(query) if sources is None else sources
        if not sources:
            return _NONE
        key = f"{'+'.join(sources)}|{query.strip().lower()}"
        hit = self._cache.get(key)
        if hit and hit[0] > time.monotonic():
            self._cache.move_to_end(key)
            logger.info("⚡ retrieval cache hit")
            return hit[1]

        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._retrieve(query, sources, key))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        try:
            # shield: a slow fetch keeps running and fills the cache for next time.
            return await asyncio.wait_for(asyncio.shield(task), timeout=wait)
        except asyncio.TimeoutError:
            logger.info("Retrieval not ready after %.1fs (%s) — answering without it",
                        wait, "+".join(sources))
            return _NONE

    async def _retrieve(self, query: str, sources: tuple[str, ...], key: str) -> dict:
        t0 = time.perf_counter()
        result = _NONE
        try:
            async with asyncio.timeout(self.s.retrieval_timeout):
                result = await self._kb_then_web(query, sources)
        except asyncio.TimeoutError:
            logger.warning("Retrieval hard timeout (%.1fs)", self.s.retrieval_timeout)
        logger.info("Retrieval %s → %s (%d chars, %.0f ms)", "+".join(sources),
                    result["source"], len(result["context"]), (time.perf_counter() - t0) * 1000)
        self._cache_put(key, result)
        return result

    async def _kb_then_web(self, query: str, sources: tuple[str, ...]) -> dict:
        kb_task = web_task = None
        try:
            if "kb" in sources:
                kb_task = asyncio.create_task(self._kb_search(query))
                if "web" in sources and self.kb_backend == "discovery":
                    done, _ = await asyncio.wait({kb_task}, timeout=_WEB_HEDGE_S)
                    if not done:
                        web_task = asyncio.create_task(self._web_search(query))
                ctx = await kb_task
                if ctx and self._is_quality(ctx, query):
                    return {"source": "kb", "context": ctx[:_MAX_CONTEXT_CHARS]}
                if "web" in sources:
                    logger.info("KB had no answer — falling back to web search")
            if "web" in sources:
                web_task = web_task or asyncio.create_task(self._web_search(query))
                ctx = await web_task
                if ctx and self._is_quality(ctx, query):
                    return {"source": "web", "context": ctx[:_MAX_CONTEXT_CHARS]}
            return _NONE
        finally:
            for task in (kb_task, web_task):
                if task and not task.done():
                    task.cancel()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_quality(ctx: str, query: str) -> bool:
        text = ctx.strip()
        if len(text) < 100:
            return False
        # English-only text for an Indic query is accepted only if substantial.
        if _INDIC_SCRIPT_RE.search(query) and not _INDIC_SCRIPT_RE.search(text):
            return len(text) >= 400
        return True

    def _cache_put(self, key: str, value: dict) -> None:
        ttl = self.s.retrieval_cache_ttl if value["source"] != "none" else 30
        self._cache[key] = (time.monotonic() + ttl, value)
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)

    async def _web_search(self, query: str) -> str:
        ctx = await self.web.search(query)
        if not ctx and _INDIC_SCRIPT_RE.search(query):
            en_query = _indic_to_english_query(query)
            if en_query:
                logger.info("Indic web fallback — English keywords: %r", en_query[:60])
                ctx = await self.web.search(en_query)
        return ctx

    async def _kb_search(self, query: str) -> str:
        try:
            if self.kb_backend == "local":
                return self.local_kb.search(query)
            return await asyncio.to_thread(self._sync_discovery_search, query)
        except Exception as exc:
            logger.error("KB search error: %s", exc)
            return ""

    def _sync_discovery_search(self, query: str) -> str:
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


