"""Web search with provider fallthrough.

Providers are tried in the configured order; a provider without credentials is
skipped silently and a provider that errors is logged and skipped. DuckDuckGo
sits at the end of the chain and needs no key, so search never hard-fails —
which matters when a demo machine has no budget for a paid search tier.

Every backend normalises to :class:`SearchResult`, so the graph is unaware of
which provider answered.
"""

from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from veritas.config import Settings, get_settings
from veritas.logging import get_logger

log = get_logger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


@dataclass(slots=True)
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    provider: str = ""
    score: float = 0.0
    published_at: str | None = None

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc.lower().removeprefix("www.")
        except ValueError:
            return ""


class SearchProvider:
    name = "base"
    requires_key = True

    def __init__(self, client: httpx.AsyncClient, api_key: str = "") -> None:
        self.client = client
        self.api_key = api_key

    @property
    def available(self) -> bool:
        return bool(self.api_key) or not self.requires_key

    async def search(self, query: str, limit: int) -> list[SearchResult]:  # pragma: no cover
        raise NotImplementedError


class TavilyProvider(SearchProvider):
    name = "tavily"

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self.client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                provider=self.name,
                score=float(item.get("score", 0.0) or 0.0),
                published_at=item.get("published_date"),
            )
            for item in data.get("results", [])
            if item.get("url")
        ]


class ExaProvider(SearchProvider):
    name = "exa"

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self.client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": self.api_key},
            json={
                "query": query,
                "numResults": limit,
                "type": "auto",
                "contents": {"text": {"maxCharacters": 1200}},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title") or "",
                snippet=(item.get("text") or "")[:1200],
                provider=self.name,
                score=float(item.get("score", 0.0) or 0.0),
                published_at=item.get("publishedDate"),
            )
            for item in data.get("results", [])
            if item.get("url")
        ]


class BraveProvider(SearchProvider):
    name = "brave"

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            params={"q": query, "count": min(limit, 20)},
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            SearchResult(
                url=item.get("url", ""),
                title=_strip_tags(item.get("title", "")),
                snippet=_strip_tags(item.get("description", "")),
                provider=self.name,
                published_at=item.get("age"),
            )
            for item in data.get("web", {}).get("results", [])
            if item.get("url")
        ]


class SearxngProvider(SearchProvider):
    """Self-hosted SearXNG — metasearch across many engines, no API key.

    The best option for this project when self-hosted: no key, no per-query
    cost, no daily quota, and it aggregates several engines so results are
    broader than any single provider. It is also the only backend here that
    cannot rate-limit you, which matters because VERITAS issues many queries
    per run.

    Requires ``search.formats: [json]`` in the instance's ``settings.yml``.
    Public instances almost always disable JSON output (it is the default), so
    a public URL will usually return HTTP 403 and be skipped — run your own.
    """

    name = "searxng"
    requires_key = False

    def __init__(self, client: httpx.AsyncClient, base_url: str = "") -> None:
        super().__init__(client, api_key="")
        self.base_url = (base_url or "").rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self.client.get(
            f"{self.base_url}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en",
                "safesearch": 0,
            },
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        )

        if resp.status_code == 403:
            log.warning(
                "SearXNG rejected the JSON API (403). Add `formats: [json]` under "
                "`search:` in settings.yml and restart the instance",
                url=self.base_url,
            )
            return []
        resp.raise_for_status()

        try:
            payload = resp.json()
        except ValueError:
            log.warning("SearXNG returned non-JSON — is `format=json` enabled?")
            return []

        results: list[SearchResult] = []
        for item in payload.get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title", "") or "",
                    snippet=(item.get("content") or "")[:1500],
                    provider=self.name,
                    score=float(item.get("score", 0.0) or 0.0),
                    published_at=item.get("publishedDate"),
                )
            )
            if len(results) >= limit:
                break
        return results


class DuckDuckGoProvider(SearchProvider):
    """Keyless fallback that parses the DuckDuckGo HTML endpoint.

    Best-effort by nature: DDG rate-limits aggressively and the markup is not a
    contract. It exists so the system degrades rather than dies when no paid
    search key is present.
    """

    name = "duckduckgo"
    requires_key = False

    _RESULT_RE = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self.client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": _USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()

        # DDG answers a rate-limited client with 200/202 and a challenge page
        # rather than an error status. Detecting that explicitly turns a silent
        # "no results" into an actionable message.
        if "result__a" not in resp.text:
            log.warning(
                "DuckDuckGo returned no result markup — likely rate-limited. "
                "Set BRAVE_API_KEY or TAVILY_API_KEY for reliable retrieval",
                status=resp.status_code,
                bytes=len(resp.text),
            )
            return []

        results: list[SearchResult] = []
        for match in self._RESULT_RE.finditer(resp.text):
            url = _unwrap_ddg(match.group("url"))
            if not url.startswith("http"):
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=_strip_tags(match.group("title")),
                    snippet=_strip_tags(match.group("snippet")),
                    provider=self.name,
                )
            )
            if len(results) >= limit:
                break
        return results


def _unwrap_ddg(href: str) -> str:
    """DDG wraps outbound links in a redirector; recover the real target."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


class SearchClient:
    """Fan-out search across the configured provider chain."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self.settings = settings or get_settings()
        self._own_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=8.0), follow_redirects=True
        )
        self._providers = self._build_chain()
        log.info(
            "search chain ready",
            providers=",".join(p.name for p in self._providers) or "none",
        )

    def _build_chain(self) -> list[SearchProvider]:
        # SearXNG takes a base URL rather than a key, but the constructor
        # signature is the same shape, so it slots into the registry unchanged.
        registry: dict[str, tuple[type[SearchProvider], str]] = {
            "searxng": (SearxngProvider, self.settings.searxng_url),
            "tavily": (TavilyProvider, self.settings.tavily_api_key),
            "exa": (ExaProvider, self.settings.exa_api_key),
            "brave": (BraveProvider, self.settings.brave_api_key),
            "duckduckgo": (DuckDuckGoProvider, ""),
        }
        chain: list[SearchProvider] = []
        for name in self.settings.search_providers:
            entry = registry.get(name)
            if entry is None:
                log.warning("unknown search provider in config", provider=name)
                continue
            cls, credential = entry
            provider = cls(self.client, credential)
            if provider.available:
                chain.append(provider)
        return chain

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

    async def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        """Return results from the first provider that answers successfully."""
        if self.settings.offline:
            return []
        if not self._providers:
            log.warning("no search providers configured")
            return []

        for provider in self._providers:
            try:
                results = await provider.search(query, limit)
            except httpx.HTTPStatusError as exc:
                log.warning(
                    "search provider rejected request",
                    provider=provider.name,
                    status=exc.response.status_code,
                )
                continue
            except Exception as exc:
                log.warning("search provider failed", provider=provider.name, error=str(exc)[:200])
                continue

            if results:
                log.debug("search ok", provider=provider.name, query=query[:60], n=len(results))
                return results

        log.warning("all search providers returned nothing", query=query[:80])
        return []

    async def search_many(
        self, queries: list[str], limit: int = 8, concurrency: int = 5
    ) -> list[SearchResult]:
        """Run several queries in parallel and merge, de-duplicating by URL."""
        sem = asyncio.Semaphore(concurrency)

        async def one(q: str) -> list[SearchResult]:
            async with sem:
                return await self.search(q, limit)

        batches = await asyncio.gather(*(one(q) for q in queries), return_exceptions=True)

        merged: dict[str, SearchResult] = {}
        for batch in batches:
            if isinstance(batch, BaseException):
                log.warning("search batch failed", error=str(batch)[:200])
                continue
            for result in batch:
                key = _canonical_url(result.url)
                existing = merged.get(key)
                if existing is None or len(result.snippet) > len(existing.snippet):
                    merged[key] = result
        return list(merged.values())

    async def aclose(self) -> None:
        if self._own_client:
            await self.client.aclose()


def _canonical_url(url: str) -> str:
    """Normalise for de-duplication: drop scheme, www, tracking params, fragment."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    keep = {
        k: v
        for k, v in parse_qs(parsed.query).items()
        if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref", "mc_"))
    }
    query = "&".join(f"{k}={quote_plus(v[0])}" for k, v in sorted(keep.items()))
    return f"{netloc}{path}" + (f"?{query}" if query else "")
