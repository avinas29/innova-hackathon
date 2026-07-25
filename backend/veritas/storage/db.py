"""SQLite persistence.

Design notes
------------
* Plain ``sqlite3`` behind a lock, with blocking calls pushed onto a worker
  thread via ``asyncio.to_thread``. This avoids an ``aiosqlite`` dependency
  while keeping the event loop unblocked.
* WAL mode so the API can read a run's events while the graph is still writing.
* The schema is deliberately Postgres-compatible: swapping the driver is a
  connection-string change, not a rewrite.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veritas.logging import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    topic         TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    finished_at   TEXT,
    config_json   TEXT NOT NULL DEFAULT '{}',
    report_json   TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    llm_calls     INTEGER NOT NULL DEFAULT 0,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    decontextualised TEXT NOT NULL DEFAULT '',
    category      TEXT NOT NULL,
    checkworthy_score REAL NOT NULL DEFAULT 0,
    verdict       TEXT NOT NULL,
    raw_confidence REAL NOT NULL DEFAULT 0,
    confidence    REAL NOT NULL DEFAULT 0,
    features_json TEXT NOT NULL DEFAULT '{}',
    rationale     TEXT NOT NULL DEFAULT '',
    retracted     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_claims_run ON claims(run_id);

CREATE TABLE IF NOT EXISTS sources (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    url           TEXT NOT NULL,
    domain        TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL DEFAULT 'WEB',
    credibility_tier  TEXT NOT NULL DEFAULT 'UNKNOWN',
    credibility_score REAL NOT NULL DEFAULT 0.45,
    degraded      INTEGER NOT NULL DEFAULT 0,
    retrieved_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_run ON sources(run_id);

CREATE TABLE IF NOT EXISTS evidence (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    claim_id      TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    url           TEXT NOT NULL DEFAULT '',
    domain        TEXT NOT NULL DEFAULT '',
    snippet       TEXT NOT NULL,
    stance        TEXT NOT NULL,
    entailment_score REAL NOT NULL DEFAULT 0,
    relevance     REAL NOT NULL DEFAULT 0,
    cluster_id    TEXT,
    is_derivative INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);

CREATE TABLE IF NOT EXISTS contradictions (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    claim_id      TEXT NOT NULL DEFAULT '',
    evidence_a    TEXT NOT NULL,
    evidence_b    TEXT NOT NULL,
    score         REAL NOT NULL DEFAULT 0,
    explanation   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_contradictions_run ON contradictions(run_id);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    ts            TEXT NOT NULL,
    node          TEXT NOT NULL DEFAULT '',
    level         TEXT NOT NULL DEFAULT 'info',
    message       TEXT NOT NULL DEFAULT '',
    payload_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);

CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


class Database:
    """Thread-safe SQLite wrapper with async-friendly helpers."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        log.debug("database ready", path=str(self.path))

    # ── low level ────────────────────────────────────────────────────────────
    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._cursor() as cur:
            cur.execute(sql, tuple(params))

    def execute_many(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        payload = [tuple(r) for r in rows]
        if not payload:
            return
        with self._cursor() as cur:
            cur.executemany(sql, payload)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(sql, tuple(params))
            return cur.fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── async wrappers ───────────────────────────────────────────────────────
    async def aexecute(self, sql: str, params: Iterable[Any] = ()) -> None:
        await asyncio.to_thread(self.execute, sql, params)

    async def aquery(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return await asyncio.to_thread(self.query, sql, params)

    async def aquery_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return await asyncio.to_thread(self.query_one, sql, params)

    # ── cache ────────────────────────────────────────────────────────────────
    def cache_get(self, key: str, ttl_seconds: int) -> str | None:
        """Read a cache entry, honouring its TTL.

        ``ttl_seconds <= 0`` means every entry is treated as stale. Reading it
        as "never expires" would be the dangerous default: a user setting TTL to
        0 to disable caching would instead get permanent caching.
        """
        if ttl_seconds <= 0:
            return None
        row = self.query_one("SELECT value, created_at FROM cache WHERE key = ?", (key,))
        if row is None:
            return None
        if (time.time() - float(row["created_at"])) > ttl_seconds:
            self.execute("DELETE FROM cache WHERE key = ?", (key,))
            return None
        return str(row["value"])

    def cache_set(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO cache(key, value, created_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at",
            (key, value, time.time()),
        )

    def cache_clear(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS n FROM cache")
        n = int(row["n"]) if row else 0
        self.execute("DELETE FROM cache")
        return n

    # ── run lifecycle ────────────────────────────────────────────────────────
    def create_run(self, run_id: str, topic: str, config: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO runs(id, topic, status, created_at, config_json) VALUES(?,?,?,?,?)",
            (run_id, topic, "PENDING", _iso_now(), json.dumps(config, default=str)),
        )

    def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        finished = _iso_now() if status in {"COMPLETED", "FAILED", "BUDGET_EXCEEDED"} else None
        self.execute(
            "UPDATE runs SET status = ?, error = ?, finished_at = COALESCE(?, finished_at) "
            "WHERE id = ?",
            (status, error, finished, run_id),
        )

    def save_report(self, run_id: str, report_json: str, usage: dict[str, int]) -> None:
        self.execute(
            "UPDATE runs SET report_json = ?, prompt_tokens = ?, completion_tokens = ?, "
            "llm_calls = ? WHERE id = ?",
            (
                report_json,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("calls", 0),
                run_id,
            ),
        )

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def list_runs(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.query(
            "SELECT id, topic, status, created_at, finished_at, prompt_tokens, "
            "completion_tokens, llm_calls FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    def delete_run(self, run_id: str) -> None:
        for table in ("events", "contradictions", "evidence", "sources", "claims"):
            self.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        self.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    # ── events (drive SSE + replay) ──────────────────────────────────────────
    def append_event(
        self,
        run_id: str,
        node: str,
        message: str,
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO events(run_id, ts, node, level, message, payload_json) "
                "VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    _iso_now(),
                    node,
                    level,
                    message,
                    json.dumps(payload or {}, default=str),
                ),
            )
            return int(cur.lastrowid or 0)

    def events_since(self, run_id: str, after_id: int = 0, limit: int = 500) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM events WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?",
            (run_id, after_id, limit),
        )

    # ── bulk persistence of a finished run ───────────────────────────────────
    def persist_report(self, report: Any) -> None:
        """Write the denormalised tables for one finished run.

        Takes a ``ResearchReport``; typed loosely to keep storage free of a
        circular import back into the domain module.
        """
        run_id = report.run_id

        self.execute_many(
            "INSERT OR REPLACE INTO claims(id, run_id, text, decontextualised, category, "
            "checkworthy_score, verdict, raw_confidence, confidence, features_json, "
            "rationale, retracted) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.id,
                    run_id,
                    c.text,
                    c.decontextualised,
                    c.category.value,
                    c.checkworthy_score,
                    c.verdict.value,
                    c.raw_confidence,
                    c.confidence,
                    json.dumps(c.features.model_dump()),
                    c.rationale,
                    int(c.retracted),
                )
                for c in report.claims
            ],
        )

        self.execute_many(
            "INSERT OR REPLACE INTO sources(id, run_id, url, domain, title, kind, "
            "credibility_tier, credibility_score, degraded, retrieved_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    s.id,
                    run_id,
                    s.url,
                    s.domain,
                    s.title,
                    s.kind.value,
                    s.credibility_tier.value,
                    s.credibility_score,
                    int(s.degraded),
                    s.retrieved_at.isoformat(),
                )
                for s in report.sources
            ],
        )

        self.execute_many(
            "INSERT OR REPLACE INTO evidence(id, run_id, claim_id, source_id, url, domain, "
            "snippet, stance, entailment_score, relevance, cluster_id, is_derivative) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    e.id,
                    run_id,
                    e.claim_id,
                    e.source_id,
                    e.url,
                    e.domain,
                    e.snippet,
                    e.stance.value,
                    e.entailment_score,
                    e.relevance,
                    e.cluster_id,
                    int(e.is_derivative),
                )
                for e in report.evidence
            ],
        )

        self.execute_many(
            "INSERT OR REPLACE INTO contradictions(id, run_id, claim_id, evidence_a, "
            "evidence_b, score, explanation) VALUES(?,?,?,?,?,?,?)",
            [
                (x.id, run_id, x.claim_id, x.evidence_a, x.evidence_b, x.score, x.explanation)
                for x in report.contradictions
            ],
        )


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_db_path(preferred: Path | str) -> Path:
    """Return a writable database path, falling back if the preferred one isn't.

    Startup must not die because a deployment target mounted a directory
    root-owned or read-only. Container platforms differ in what they make
    writable, and the failure mode is a bare ``sqlite3.OperationalError:
    unable to open database file`` with no indication of which path failed.

    Candidates are tried in order: the configured path, ``/tmp``, then the
    system temp directory. Falling back loses persistence across restarts, so
    it is logged as a warning rather than silently accepted.
    """
    import tempfile

    preferred = Path(preferred)
    candidates = [
        preferred,
        Path("/tmp") / preferred.name,
        Path(tempfile.gettempdir()) / preferred.name,
    ]

    for index, candidate in enumerate(candidates):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            # mkdir succeeding does not prove writability — the directory may
            # already exist and be owned by another user. Probe it.
            probe = candidate.parent / f".veritas-write-test-{os.getpid()}"
            probe.touch()
            probe.unlink()
        except OSError as exc:
            log.warning(
                "database location not writable",
                path=str(candidate),
                error=f"{type(exc).__name__}: {exc}",
            )
            continue

        if index > 0:
            log.warning(
                "falling back to a writable database location — data will NOT "
                "persist across restarts",
                requested=str(preferred),
                using=str(candidate),
            )
        return candidate

    raise RuntimeError(
        f"no writable database location found. Tried: "
        f"{', '.join(str(c) for c in candidates)}. "
        "Set VERITAS_DB_PATH to a writable path."
    )


_DB: Database | None = None
_DB_LOCK = threading.Lock()


def get_db(path: Path | str | None = None) -> Database:
    """Process-wide database handle."""
    global _DB
    with _DB_LOCK:
        if _DB is None:
            if path is None:
                from veritas.config import get_settings

                path = get_settings().db_file
            _DB = Database(resolve_db_path(path))
        return _DB


def reset_db() -> None:
    """Close and drop the singleton — used by tests between cases."""
    global _DB
    with _DB_LOCK:
        if _DB is not None:
            _DB.close()
        _DB = None
