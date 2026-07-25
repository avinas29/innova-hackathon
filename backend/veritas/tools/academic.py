"""Keyless primary-source connectors: arXiv, Semantic Scholar, Wikipedia.

Professional fact-checkers prioritise primary sources over secondary reporting,
and the Anthropic multi-agent postmortem specifically flags agents drifting to
"SEO-optimized content farms over authoritative but less highly-ranked sources".
Giving the graph direct, unranked access to scholarly and reference APIs is a
structural fix for that bias — it does not depend on a web-search ranker
surfacing the right thing.

All three APIs are usable without credentials.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from xml.etree import ElementTree

import httpx

from veritas.logging import get_logger
from veritas.tools.search import SearchResult

log = get_logger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_WIKI_API = "https://en.wikipedia.org/w/api.php"

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass(slots=True)
class Paper:
    title: str
    abstract: str
    url: str
    authors: list[str]
    published: str = ""
    venue: str = ""
    citation_count: int = 0
    source: str = ""

    def to_search_result(self) -> SearchResult:
        author_line = ", ".join(self.authors[:4]) + (" et al." if len(self.authors) > 4 else "")
        snippet = f"{author_line}. {self.abstract}".strip()
        return SearchResult(
            url=self.url,
            title=self.title,
            snippet=snippet[:1500],
            provider=self.source,
            score=min(1.0, self.citation_count / 200) if self.citation_count else 0.5,
            published_at=self.published,
        )


class AcademicClient:
    """Fetches scholarly and encyclopedic primary sources."""

    def __init__(
        self, client: httpx.AsyncClient | None = None, offline: bool | None = None
    ) -> None:
        if offline is None:
            from veritas.config import get_settings

            offline = get_settings().offline
        self.offline = offline
        self._own_client = client is None
        # Wikimedia's API policy rejects generic agents with 403. The required
        # shape is Tool/Version (contact URL) plus the HTTP library identifier.
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=8.0),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "VERITAS/1.0 (https://github.com/veritas-research/veritas; "
                    "fact-verification research agent) python-httpx"
                ),
                "Accept": "application/json",
            },
        )

    # ── arXiv ────────────────────────────────────────────────────────────────
    async def arxiv(self, query: str, limit: int = 6) -> list[Paper]:
        if self.offline:
            return []
        try:
            resp = await self.client.get(
                _ARXIV_API,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance",
                },
            )
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.text)
        except Exception as exc:
            log.warning("arxiv query failed", error=str(exc)[:200])
            return []

        papers: list[Paper] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            title = _text(entry, "atom:title")
            summary = _text(entry, "atom:summary")
            link = _text(entry, "atom:id")
            published = _text(entry, "atom:published")[:10]
            authors = [
                _text(a, "atom:name")
                for a in entry.findall("atom:author", _ATOM_NS)
                if _text(a, "atom:name")
            ]
            if title and link:
                papers.append(
                    Paper(
                        title=title,
                        abstract=summary,
                        url=link,
                        authors=authors,
                        published=published,
                        venue="arXiv",
                        source="arxiv",
                    )
                )
        return papers

    # ── Semantic Scholar ─────────────────────────────────────────────────────
    async def semantic_scholar(self, query: str, limit: int = 6) -> list[Paper]:
        if self.offline:
            return []
        try:
            resp = await self.client.get(
                _S2_API,
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,url,year,venue,citationCount,authors",
                },
            )
            if resp.status_code == 429:
                log.debug("semantic scholar rate limited")
                return []
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("semantic scholar query failed", error=str(exc)[:200])
            return []

        papers: list[Paper] = []
        for item in data.get("data", []) or []:
            if not item.get("title"):
                continue
            papers.append(
                Paper(
                    title=item["title"],
                    abstract=item.get("abstract") or "",
                    url=item.get("url") or "",
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    published=str(item.get("year") or ""),
                    venue=item.get("venue") or "",
                    citation_count=int(item.get("citationCount") or 0),
                    source="semanticscholar",
                )
            )
        return [p for p in papers if p.url]

    # ── Wikipedia ────────────────────────────────────────────────────────────
    async def wikipedia(self, query: str, limit: int = 3) -> list[SearchResult]:
        if self.offline:
            return []
        try:
            resp = await self.client.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
        except Exception as exc:
            log.warning("wikipedia search failed", error=str(exc)[:200])
            return []

        results: list[SearchResult] = []
        for hit in hits:
            title = hit.get("title", "")
            if not title:
                continue
            results.append(
                SearchResult(
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    title=title,
                    snippet=re.sub(r"<[^>]+>", "", hit.get("snippet", "")),
                    provider="wikipedia",
                )
            )
        return results

    async def search_scholarly(self, query: str, limit: int = 6) -> list[SearchResult]:
        """Query arXiv and Semantic Scholar together, merged and de-duplicated."""
        if self.offline:
            return []
        arxiv_task = self.arxiv(query, limit)
        s2_task = self.semantic_scholar(query, limit)
        batches = await asyncio.gather(arxiv_task, s2_task, return_exceptions=True)

        seen: set[str] = set()
        merged: list[SearchResult] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                continue
            for paper in batch:
                key = re.sub(r"\W+", "", paper.title.lower())[:80]
                if key in seen:
                    continue
                seen.add(key)
                merged.append(paper.to_search_result())
        return merged

    async def aclose(self) -> None:
        if self._own_client:
            await self.client.aclose()


def _text(node: ElementTree.Element, path: str) -> str:
    found = node.find(path, _ATOM_NS)
    if found is None or found.text is None:
        return ""
    return re.sub(r"\s+", " ", found.text).strip()
