"""
llm/web_search.py
Web search for real-time questions.

  • BRAVE_SEARCH_API_KEY set → official Brave Search API (fast, reliable, ToS-safe;
    recommended for production). Scrapers are used only if the API fails.
  • Otherwise DuckDuckGo (ddgs) and Bing HTML are queried *concurrently* and
    the first non-empty result wins (measured: ~0.3 s vs 2–4.5 s for the old
    DDG → Brave → Bing sequential cascade; Brave's HTML page now returns 429
    to scrapers, so only its official API is used).
  • Gold / silver price questions also scrape goodreturns.in in parallel and
    prefer it when it answers.

All HTTP goes through the shared pooled client (core.http).
"""
from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from core.http import BROWSER_HEADERS, get_http_client, request_with_retry

logger = get_logger("llm.web_search")

try:
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover
    DDGS = None
    logger.warning("ddgs not installed — DuckDuckGo search disabled (pip install ddgs)")

try:
    import lxml  # noqa: F401
    _PARSER = "lxml"
except ImportError:  # pragma: no cover
    _PARSER = "html.parser"

_GOLD_RE = re.compile(
    r"\b(gold|silver|sona|chandi|sone)\b"
    r"|सोना|सोने|चांदी|चाँदी|தங்கம்|வெள்ளி|బంగారం|వెండి|ಚಿನ್ನ|ಬೆಳ್ಳಿ|সোনা|রুপা|સોનુ|ચાંદી",
    re.IGNORECASE,
)
_GOODRETURNS_URL = "https://www.goodreturns.in/gold-rates/"
_MAX_RESULTS = 5
_ENGINE_TIMEOUT = 3.5

# ddgs is synchronous; a small dedicated pool bounds the threads it can use.
_DDG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ddg")


def _format_results(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results[:_MAX_RESULTS], 1):
        title = r.get("title", "")
        body = r.get("body") or r.get("description") or ""
        parts.append(f"RESULT {i}\nTitle: {title}\nContent: {body}")
    return "\n\n".join(parts)


class WebSearchService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()

    async def warmup(self) -> None:
        """Pre-open TLS connections to the search hosts."""
        client = get_http_client()
        hosts = (["https://api.search.brave.com"] if self.s.brave_search_api_key
                 else ["https://www.bing.com"])
        await asyncio.gather(*(client.head(h, headers=BROWSER_HEADERS, timeout=3)
                               for h in hosts), return_exceptions=True)

    # ── public ────────────────────────────────────────────────────────────────

    async def search(self, query: str) -> str:
        """Plain-text context for `query`, or '' — never raises."""
        try:
            if _GOLD_RE.search(query):
                gold = asyncio.create_task(self._gold_rates())
                general = asyncio.create_task(self._general(query))
                try:
                    result = await gold
                    if result:
                        return result
                    return await general
                finally:
                    general.cancel()
            return await self._general(query)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("web search error: %r", exc)
            return ""

    async def _general(self, query: str) -> str:
        if self.s.brave_search_api_key:
            results = await self._brave_api(query)
            if results:
                return _format_results(results)
        return _format_results(await self._race_scrapers(query))

    # ── engines ───────────────────────────────────────────────────────────────

    async def _race_scrapers(self, query: str) -> list[dict]:
        engines = {
            asyncio.create_task(self._ddg(query)): "ddg",
            asyncio.create_task(self._bing_html(query)): "bing",
        }
        pending = set(engines)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, timeout=_ENGINE_TIMEOUT, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    break
                for task in done:
                    results = task.result() if not task.exception() else []
                    if results:
                        logger.info("web: %s won with %d results", engines[task], len(results))
                        return results
            logger.warning("web: no engine returned results for %r", query[:60])
            return []
        finally:
            for task in pending:
                task.cancel()

    async def _brave_api(self, query: str) -> list[dict]:
        client = get_http_client()
        try:
            resp = await request_with_retry(
                lambda: client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": _MAX_RESULTS},
                    headers={"X-Subscription-Token": self.s.brave_search_api_key,
                             "Accept": "application/json"},
                    timeout=_ENGINE_TIMEOUT),
                label="brave-api")
            resp.raise_for_status()
            return [{"title": r.get("title", ""), "body": r.get("description", "")}
                    for r in resp.json().get("web", {}).get("results", [])]
        except Exception as exc:
            logger.warning("Brave Search API failed: %r", exc)
            return []

    async def _ddg(self, query: str) -> list[dict]:
        if DDGS is None:
            return []

        def run() -> list[dict]:
            with DDGS(timeout=int(_ENGINE_TIMEOUT)) as ddgs:
                return list(ddgs.text(query, max_results=_MAX_RESULTS))
        try:
            return await asyncio.get_running_loop().run_in_executor(_DDG_POOL, run)
        except Exception as exc:
            logger.debug("ddg failed: %r", exc)
            return []

    async def _fetch_html(self, url: str) -> str:
        resp = await get_http_client().get(url, headers=BROWSER_HEADERS, timeout=_ENGINE_TIMEOUT)
        resp.raise_for_status()
        return resp.text

    async def _bing_html(self, query: str) -> list[dict]:
        try:
            html = await self._fetch_html(
                f"https://www.bing.com/search?q={quote_plus(query)}&count={_MAX_RESULTS}")
            soup = BeautifulSoup(html, _PARSER)
            out = []
            for item in soup.select("li.b_algo")[:_MAX_RESULTS]:
                title = item.select_one("h2")
                desc = item.select_one(".b_caption p, p")
                entry = {"title": title.get_text(strip=True) if title else "",
                         "body": desc.get_text(strip=True) if desc else ""}
                if entry["title"] or entry["body"]:
                    out.append(entry)
            return out
        except Exception as exc:
            logger.debug("bing html failed: %r", exc)
            return []

    async def _gold_rates(self) -> str:
        try:
            soup = BeautifulSoup(await self._fetch_html(_GOODRETURNS_URL), _PARSER)
        except Exception as exc:
            logger.warning("goodreturns.in fetch failed: %r", exc)
            return ""
        cards = [c.get_text(" ", strip=True) for c in soup.select(".gr-price-card")]
        if not cards:
            cards = [el.get_text(strip=True) for el in
                     soup.select(".gold-rate-price, .price-value, [class*='price']")
                     if "₹" in el.get_text()]
        rows: list[str] = []
        headers: list[str] = []
        table = soup.select_one(".gr-table") or soup.select_one("table")
        if table:
            headers = [th.get_text(strip=True) for th in table.select("thead th")]
            for row in table.select("tbody tr"):
                cells = [td.get_text(strip=True) for td in row.select("td")]
                if any("₹" in c for c in cells):
                    rows.append(" | ".join(cells))
        if not cards and not rows:
            logger.warning("goodreturns.in: no rate data found (page layout changed?)")
            return ""
        lines = ["Live gold rates in India (goodreturns.in)"]
        if cards:
            lines += ["Today's rate per gram:", *(f"  {c}" for c in cards[:4])]
        if rows:
            lines.append(f"City-wise [{' | '.join(headers) or 'City | 24K | 22K'}]:")
            lines += [f"  {r}" for r in rows[:8]]
        return "\n".join(lines)
