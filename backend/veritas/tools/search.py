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

    @property
    def search_depth(self) -> str:
        from veritas.config import get_settings

        return get_settings().tavily_search_depth

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self.client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": limit,
                "search_depth": self.search_depth,
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


def _is_unresolvable(exc: Exception) -> bool:
    """True when the host does not exist, as opposed to being asleep.

    These are different failures with different fixes, and conflating them
    wastes a run: a sleeping instance is worth waiting 75s for, a hostname that
    does not exist never will be. Retrying DNS burned ~17s per query and
    produced a report with two sources instead of twenty-four.
    """
    text = str(exc).lower()
    return (
        "name or service not known" in text
        or "nodename nor servname" in text
        or "temporary failure in name resolution" in text
        or "getaddrinfo" in text
    )


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

    # A free PaaS instance sleeps when idle and takes ~50s to wake. The shared
    # client timeout is far shorter, so the first query after a quiet spell
    # would always fail and the provider would be skipped for the whole run.
    COLD_START_TIMEOUT = 75.0
    WAKE_ATTEMPTS = 3

    def __init__(self, client: httpx.AsyncClient, base_url: str = "") -> None:
        super().__init__(client, api_key="")
        self.base_url = _normalise_base_url(base_url)
        self._disabled = False
        self._awake = False
        self._wake_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return bool(self.base_url) and not self._disabled

    async def _ensure_awake(self) -> None:
        """Wake a sleeping instance once, with every other caller waiting.

        This is the difference between 2 sources and 24. A run fans out to
        several researchers, each issuing multiple queries, so a cold instance
        gets hit by ~6 concurrent requests — and *all* of them fail together
        before it has finished booting. The run then silently falls back to
        arXiv and produces a thin report.

        The lock collapses that burst into a single wake request that the rest
        await, so the instance boots once and every query lands on a live
        service.
        """
        if self._awake:
            return

        async with self._wake_lock:
            if self._awake:  # another caller woke it while we queued
                return

            for attempt in range(self.WAKE_ATTEMPTS):
                try:
                    resp = await self.client.get(
                        f"{self.base_url}/healthz", timeout=self.COLD_START_TIMEOUT
                    )
                    if resp.status_code < 500:
                        if attempt:
                            log.info("SearXNG awake", url=self.base_url, attempts=attempt + 1)
                        self._awake = True
                        return
                except Exception as exc:
                    if _is_unresolvable(exc):
                        log.error(
                            "SEARXNG_URL does not resolve — search is disabled for this "
                            "run. Set it to the search service's PUBLIC url (e.g. "
                            "https://veritas-searxng.onrender.com). A Docker Compose "
                            "service name like http://searxng:8080 only resolves inside "
                            "Compose, never on a hosting platform",
                            url=self.base_url,
                        )
                        # Disable via an explicit flag, not by blanking the URL:
                        # search() would then build "" + "/search" = "/search",
                        # which httpx rejects as protocol-less — turning a clear
                        # DNS error into a confusing UnsupportedProtocol one.
                        self._disabled = True
                        return
                    log.info(
                        "waking SearXNG — a sleeping free instance takes ~50s",
                        attempt=attempt + 1,
                        error=type(exc).__name__,
                    )
                await asyncio.sleep(2.0 * (attempt + 1))

            # Give up on the probe but still attempt the search: /healthz may be
            # blocked while the search endpoint itself works.
            log.warning("SearXNG did not answer /healthz — trying search anyway", url=self.base_url)
            self._awake = True

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        if self._disabled or not self.base_url:
            return []
        await self._ensure_awake()
        if self._disabled:  # the wake attempt may have just disabled us
            return []
        try:
            return await self._search_once(query, limit)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            # Still booting despite the wake probe — one more try, then give up
            # to the next provider in the chain.
            log.info("SearXNG connection failed — retrying once", url=self.base_url)
            await asyncio.sleep(5.0)
            return await self._search_once(query, limit)

    async def _search_once(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self.client.get(
            f"{self.base_url}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en",
                "safesearch": 0,
            },
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
            timeout=self.COLD_START_TIMEOUT,
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


async def _suppress(coro) -> None:
    """Run a background warm-up; never let its failure surface."""
    try:
        await coro
    except Exception as exc:
        log.debug("warm-up failed", error=str(exc)[:120])


async def warm_searxng(
    base_url: str, timeout: float = 90.0, poll_interval: float = 3.0
) -> bool:
    """Wake a sleeping SearXNG instance and wait until it answers.

    Free-tier hosting spins an idle service down; the first request then takes
    ~50 seconds while the container cold-starts. A research run fires several
    searches concurrently, so without this they *all* hit the cold instance at
    once, time out together, and the run silently falls back to scholarly
    sources — producing two sources per question instead of twenty.

    Waiting once, up front, is far cheaper than every query failing. Called at
    app startup (fire-and-forget, so the wake overlaps with the user reading the
    page) and again at the start of each run (awaited, bounded).
    """
    from veritas.config import get_settings

    # Respect the hard network kill-switch, whatever the caller passed.
    if get_settings().offline:
        return False

    base_url = _normalise_base_url(base_url)
    if not base_url:
        return False

    deadline = asyncio.get_running_loop().time() + timeout
    attempt = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
        while asyncio.get_running_loop().time() < deadline:
            attempt += 1
            try:
                resp = await client.get(f"{base_url}/healthz")
                if resp.status_code == 200:
                    if attempt > 1:
                        log.info("SearXNG is awake", url=base_url, attempts=attempt)
                    return True
            except Exception as exc:
                log.debug(
                    "SearXNG still waking", url=base_url, attempt=attempt, error=type(exc).__name__
                )
            await asyncio.sleep(poll_interval)

    log.warning(
        "SearXNG did not wake within the timeout — search will fall back",
        url=base_url,
        seconds=timeout,
    )
    return False


def _normalise_base_url(raw: str) -> str:
    """Accept a bare hostname as well as a full URL.

    Render's ``fromService`` wiring yields a hostname with no scheme
    (``veritas-searxng.onrender.com``), which httpx rejects outright. Defaulting
    to https keeps the blueprint declarative instead of requiring the URL to be
    pasted by hand.
    """
    raw = (raw or "").strip().rstrip("/").rstrip(".")
    if not raw:
        return ""

    # A hostname needs a dot or an explicit port. Without one this is almost
    # certainly the wrong value pasted in — a secret, a service name, a token.
    # Prepending https:// to it produces a host that cannot resolve, and the
    # resulting DNS error says nothing about the real mistake.
    bare = raw.split("://", 1)[-1]
    if "." not in bare and ":" not in bare and bare not in {"localhost"}:
        log.error(
            "SEARXNG_URL does not look like a URL — search disabled. Expected "
            "something like https://your-searxng.onrender.com. Check you have not "
            "pasted a secret or a service name into this variable",
            value=raw[:60],
        )
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    # Local hosts are almost never TLS-terminated; anything else on a PaaS is.
    scheme = "http" if raw.startswith(("localhost", "127.0.0.1", "0.0.0.0")) else "https"
    return f"{scheme}://{raw}"


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
        self._warm_tasks: set[asyncio.Task] = set()
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

    def warm_up(self) -> None:
        """Start waking sleep-prone providers, without blocking the caller.

        Fired when a run's context is built so the ~50s cold start overlaps
        with planning — which takes a few seconds anyway — instead of being
        paid in full by the first search.
        """
        for provider in self._providers:
            ensure = getattr(provider, "_ensure_awake", None)
            if ensure is None:
                continue
            task = asyncio.create_task(_suppress(ensure()))
            # Hold a reference: a bare create_task can be garbage-collected
            # mid-flight, which would silently cancel the wake.
            self._warm_tasks.add(task)
            task.add_done_callback(self._warm_tasks.discard)

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
