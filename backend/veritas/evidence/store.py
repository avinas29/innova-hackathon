"""Embeddings and vector search.

Why not FAISS: a run holds on the order of 10^2–10^3 vectors. At that scale a
single NumPy matrix multiply beats an index build, with none of the wheel
fragility FAISS has on Python 3.13. The interface below is the same shape an
ANN index would expose, so swapping in FAISS or pgvector later touches one
class.

The offline embedder is a real algorithm — feature hashing with signed buckets,
the classic "hashing trick" — not a stub. It is weaker than a learned model but
deterministic and dependency-free, which is what CI needs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from veritas.config import Settings, get_settings
from veritas.llm.client import tokenize
from veritas.logging import get_logger

log = get_logger(__name__)

_HASH_DIM = 512


class Embedder:
    """Base interface: text in, unit-normalised vectors out."""

    dim: int = 0
    name: str = "base"

    async def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embedder (no network, no model).

    Maps each token to a bucket via BLAKE2b, accumulates signed sublinear term
    weights, then L2-normalises. Captures lexical overlap well enough to drive
    near-duplicate clustering, which is what the offline path needs it for.
    """

    name = "hashing"

    def __init__(self, dim: int = _HASH_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])


class OpenAIEmbedder(Embedder):
    """Embeddings over the OpenAI wire format, with a hashing fallback.

    Also serves Gemini via its OpenAI-compatible ``/embeddings`` route — pass the
    compatibility ``base_url``. Dimensionality is read from the first response
    rather than hardcoded, since it varies across providers and model versions.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        name: str = "openai",
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install 'veritas[openai]' for embeddings") from exc
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2
        )
        self.model = model
        self.name = name
        self.dim = 1536  # provisional; corrected from the first live response
        self._fallback = HashingEmbedder()

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        cleaned = [(t or " ").replace("\n", " ")[:8000] for t in texts]
        vectors: list[list[float]] = []
        batch_size = 128

        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            try:
                resp = await self._client.embeddings.create(model=self.model, input=chunk)
            except Exception as exc:
                log.warning(
                    "embedding request failed — falling back to hashing embedder",
                    error=str(exc)[:200],
                )
                return await self._fallback.embed(texts)
            vectors.extend(item.embedding for item in resp.data)

        matrix = np.asarray(vectors, dtype=np.float32)
        self.dim = int(matrix.shape[1])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Pick an embedder for the active provider.

    Anthropic ships no embedding endpoint, so an Anthropic-only setup falls back
    to hashing. That is a real quality cost on semantic dedup — set an OpenAI or
    Gemini key alongside it if that matters.
    """
    settings = settings or get_settings()
    provider = settings.resolved_provider

    if provider == "openai" and settings.openai_api_key:
        return OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)

    if provider == "gemini" and settings.gemini_api_key:
        from veritas.config import GEMINI_BASE_URL

        return OpenAIEmbedder(
            settings.gemini_api_key,
            settings.embedding_model_gemini,
            base_url=GEMINI_BASE_URL,
            name="gemini",
        )

    log.info("using deterministic hashing embedder", provider=provider)
    return HashingEmbedder()


# ─────────────────────────────────────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class VectorHit:
    id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """In-memory cosine store with optional SQLite persistence.

    Vectors are L2-normalised on insert, so cosine similarity is a dot product
    and the whole search is one ``matrix @ query`` call.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or build_embedder()
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._matrix: np.ndarray | None = None
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._ids)

    async def add(
        self, ids: list[str], texts: list[str], metadata: list[dict] | None = None
    ) -> None:
        if not ids:
            return
        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length")
        meta = metadata or [{} for _ in ids]

        vectors = await self.embedder.embed(texts)
        async with self._lock:
            self._ids.extend(ids)
            self._texts.extend(texts)
            self._meta.extend(meta)
            self._matrix = (
                vectors if self._matrix is None else np.vstack([self._matrix, vectors])
            )

    async def search(self, query: str, k: int = 8) -> list[VectorHit]:
        if self._matrix is None or not self._ids:
            return []
        query_vec = (await self.embedder.embed([query]))[0]
        scores = self._matrix @ query_vec
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            VectorHit(
                id=self._ids[i],
                score=float(scores[i]),
                text=self._texts[i],
                metadata=self._meta[i],
            )
            for i in top
        ]

    async def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """Pairwise cosine similarity for a list of texts."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = await self.embedder.embed(texts)
        return vectors @ vectors.T

    def persist(self, path: str) -> None:
        """Save to a ``.npz`` sidecar so a run can be reloaded for replay."""
        if self._matrix is None:
            return
        np.savez_compressed(
            path,
            matrix=self._matrix,
            ids=np.array(self._ids, dtype=object),
            texts=np.array(self._texts, dtype=object),
            meta=np.array([json.dumps(m, default=str) for m in self._meta], dtype=object),
        )

    @classmethod
    def load(cls, path: str, embedder: Embedder | None = None) -> VectorStore:
        store = cls(embedder)
        data = np.load(path, allow_pickle=True)
        store._matrix = data["matrix"]
        store._ids = list(data["ids"])
        store._texts = list(data["texts"])
        store._meta = [json.loads(m) for m in data["meta"]]
        return store


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def chunk_text(text: str, target_chars: int = 900, overlap: int = 150) -> list[str]:
    """Split on paragraph boundaries, packing to roughly ``target_chars``.

    Paragraph-aligned chunks keep a claim and its qualifying clause together;
    fixed-width windows routinely sever them, which produces evidence that
    entails the opposite of what the source says.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > target_chars * 2:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_hard_split(para, target_chars, overlap))
            continue

        if not buffer:
            buffer = para
        elif len(buffer) + len(para) + 2 <= target_chars:
            buffer += "\n\n" + para
        else:
            chunks.append(buffer)
            tail = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
            buffer = (tail + "\n\n" + para).strip() if tail else para

    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if len(c.strip()) > 40]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]
