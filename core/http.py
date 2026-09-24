"""
core/http.py
One pooled async HTTP client for every REST integration (Sarvam, ElevenLabs,
web scraping, …) plus a small retry helper that honours 429 Retry-After.

Reusing a single client keeps TCP + TLS connections warm (HTTP/2 when the
server supports it), saving 50–150 ms per request versus opening a new session.
"""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable

import httpx

from config.logging_config import get_logger

logger = get_logger("core.http")

_client: httpx.AsyncClient | None = None

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        try:
            import h2  # noqa: F401  (optional)
            http2 = True
        except ImportError:
            http2 = False
        _client = httpx.AsyncClient(
            http2=http2,
            timeout=httpx.Timeout(10.0, connect=3.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=40,
                                keepalive_expiry=120),
            follow_redirects=True,
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def request_with_retry(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    attempts: int = 2,
    base_delay: float = 0.25,
    max_delay: float = 2.0,
    label: str = "http",
) -> httpx.Response:
    """
    Run `send()` and retry on transient failures (connection errors, 408/429/5xx).
    Delays are short on purpose: in a voice loop a late answer is a wrong answer,
    so we fail over to a fallback provider rather than wait long.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = await send()
            if resp.status_code not in RETRYABLE_STATUS or attempt == attempts:
                return resp
            retry_after = resp.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() \
                else base_delay * 2 ** (attempt - 1)
            logger.warning("%s: HTTP %d — retry %d/%d in %.2fs",
                           label, resp.status_code, attempt, attempts - 1, min(delay, max_delay))
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("%s: %s — retry %d/%d", label, type(exc).__name__, attempt, attempts - 1)
        await asyncio.sleep(min(delay, max_delay) + random.uniform(0, 0.05))
    raise last_exc or RuntimeError(f"{label}: retries exhausted")  # pragma: no cover
