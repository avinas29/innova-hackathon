"""Page fetching and main-content extraction.

Failure here is expected and routine — paywalls, bot walls, timeouts. A failed
fetch must never fail a run: we fall back to the search snippet and mark the
source ``degraded`` so the confidence model can discount it.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from veritas.config import Settings, get_settings
from veritas.logging import get_logger

log = get_logger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
_MAX_BYTES = 3_000_000
_MAX_CHARS = 60_000

_SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".gz", ".tar", ".mp4", ".mp3", ".png", ".jpg",
    ".jpeg", ".gif", ".svg", ".webp", ".ico", ".exe", ".dmg",
)


@dataclass(slots=True)
class FetchedPage:
    url: str
    title: str = ""
    text: str = ""
    ok: bool = False
    status: int = 0
    error: str = ""

    @property
    def degraded(self) -> bool:
        return not self.ok or len(self.text) < 200


class ContentFetcher:
    """Concurrency-limited fetcher with SQLite-backed caching."""

    def __init__(self, settings: Settings | None = None, concurrency: int = 8) -> None:
        self.settings = settings or get_settings()
        self._sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=6.0),
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            limits=httpx.Limits(max_connections=concurrency * 2),
        )

    async def fetch(self, url: str) -> FetchedPage:
        if self.settings.offline:
            return FetchedPage(url=url, ok=False, error="offline mode")
        if not _fetchable(url):
            return FetchedPage(url=url, ok=False, error="unsupported url or content type")

        cache_key = "page:" + hashlib.sha256(url.encode()).hexdigest()
        cached = await asyncio.to_thread(self._cache_get, cache_key)
        if cached is not None:
            title, _, text = cached.partition("\x00")
            return FetchedPage(url=url, title=title, text=text, ok=bool(text), status=200)

        async with self._sem:
            try:
                resp = await self.client.get(url)
            except Exception as exc:
                log.debug("fetch failed", url=url[:90], error=type(exc).__name__)
                return FetchedPage(url=url, ok=False, error=str(exc)[:200])

        if resp.status_code >= 400:
            return FetchedPage(url=url, ok=False, status=resp.status_code, error=f"HTTP {resp.status_code}")

        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return FetchedPage(url=url, ok=False, status=resp.status_code, error=f"content-type {content_type}")

        raw = resp.content[:_MAX_BYTES]
        try:
            html_text = raw.decode(resp.encoding or "utf-8", errors="replace")
        except (LookupError, UnicodeDecodeError):
            html_text = raw.decode("utf-8", errors="replace")

        title, text = extract_main_content(html_text)
        text = text[:_MAX_CHARS]

        if text:
            await asyncio.to_thread(self._cache_set, cache_key, f"{title}\x00{text}")

        return FetchedPage(
            url=url, title=title, text=text, ok=bool(text), status=resp.status_code
        )

    async def fetch_many(self, urls: list[str]) -> dict[str, FetchedPage]:
        pages = await asyncio.gather(*(self.fetch(u) for u in urls), return_exceptions=True)
        out: dict[str, FetchedPage] = {}
        for url, page in zip(urls, pages, strict=True):
            if isinstance(page, BaseException):
                out[url] = FetchedPage(url=url, ok=False, error=str(page)[:200])
            else:
                out[url] = page
        return out

    def _cache_get(self, key: str) -> str | None:
        if not self.settings.cache_enabled:
            return None
        from veritas.storage.db import get_db

        try:
            return get_db().cache_get(key, self.settings.cache_ttl_seconds)
        except Exception:
            return None

    def _cache_set(self, key: str, value: str) -> None:
        if not self.settings.cache_enabled:
            return
        from veritas.storage.db import get_db

        try:
            get_db().cache_set(key, value)
        except Exception:
            pass

    async def aclose(self) -> None:
        await self.client.aclose()


def _fetchable(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return not parsed.path.lower().endswith(_SKIP_EXTENSIONS)


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript|svg|nav|footer|header|aside)[^>]*>.*?</\1>", re.DOTALL | re.I)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.I)
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def extract_main_content(html_text: str) -> tuple[str, str]:
    """Return ``(title, main_text)``.

    Uses trafilatura when importable — it is markedly better at stripping
    boilerplate — and falls back to a conservative regex strip otherwise, so
    extraction still works if the optional dependency is missing.
    """
    title = ""
    match = _TITLE_RE.search(html_text)
    if match:
        title = _clean(_TAG_RE.sub("", match.group(1)))[:300]

    try:
        import trafilatura

        extracted = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        if extracted and len(extracted) > 150:
            meta_title = _trafilatura_title(html_text)
            return (meta_title or title), _clean(extracted)
    except ImportError:
        log.debug("trafilatura unavailable — using regex extraction")
    except Exception as exc:
        log.debug("trafilatura failed", error=str(exc)[:120])

    stripped = _SCRIPT_RE.sub(" ", html_text)
    stripped = re.sub(r"<(p|div|br|li|h[1-6]|tr)[^>]*>", "\n", stripped, flags=re.I)
    text = _TAG_RE.sub(" ", stripped)
    return title, _clean(_unescape(text))


def _trafilatura_title(html_text: str) -> str:
    try:
        import trafilatura

        meta = trafilatura.extract_metadata(html_text)
        return (getattr(meta, "title", "") or "")[:300]
    except Exception:
        return ""


def _unescape(text: str) -> str:
    import html as html_mod

    return html_mod.unescape(text)


def _clean(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()
