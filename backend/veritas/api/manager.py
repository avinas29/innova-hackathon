"""Run lifecycle management and event fan-out.

Holds in-flight runs, brokers SSE subscribers, and persists everything to
SQLite so a client that connects late — or reconnects after a drop — can replay
the run from the beginning rather than seeing a dead stream. That replay path
is what makes the live demo robust to a flaky conference network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from veritas.config import get_settings
from veritas.logging import get_logger
from veritas.schemas import ResearchReport, RunEvent, RunStatus
from veritas.storage.db import get_db

log = get_logger(__name__)

_MAX_BUFFER = 2000


@dataclass
class RunHandle:
    run_id: str
    topic: str
    status: RunStatus = RunStatus.PENDING
    task: asyncio.Task | None = None
    report: ResearchReport | None = None
    error: str = ""
    events: list[RunEvent] = field(default_factory=list)
    # Sequence of events[0]. The buffer is trimmed, so a list index is not a
    # stable id — a reconnect must resume by sequence, not position.
    first_seq: int = 0
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def finished(self) -> bool:
        return self.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.BUDGET_EXCEEDED,
        }


class RunManager:
    """Owns every run in the process."""

    def __init__(self) -> None:
        self._runs: dict[str, RunHandle] = {}
        self._lock = asyncio.Lock()

    # ── queries ──────────────────────────────────────────────────────────────
    def get(self, run_id: str) -> RunHandle | None:
        return self._runs.get(run_id)

    def list_active(self) -> list[RunHandle]:
        return [h for h in self._runs.values() if not h.finished]

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self, topic: str, run_id: str | None = None) -> RunHandle:
        from veritas.graph.build import run_research

        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        handle = RunHandle(run_id=run_id, topic=topic, status=RunStatus.RUNNING)

        async with self._lock:
            self._runs[run_id] = handle

        settings = get_settings()
        db = get_db()
        await asyncio.to_thread(
            db.create_run, run_id, topic, {"provider": settings.resolved_provider}
        )
        await asyncio.to_thread(db.set_run_status, run_id, "RUNNING")

        def sink(event: RunEvent) -> None:
            self._publish(handle, event)

        async def runner() -> None:
            # Let the POST response flush before doing any setup work.
            #
            # build_context() constructs five service objects synchronously
            # (HTTP clients, embedder, vector store). On a small instance that
            # blocks the event loop for seconds, and because the task starts
            # eagerly it delayed the 202 response by 9.4s in production. One
            # yield is enough: the response is already queued.
            await asyncio.sleep(0)
            try:
                report = await run_research(topic, run_id, settings, event_sink=sink)
                handle.report = report
                handle.status = report.status

                await asyncio.to_thread(
                    db.save_report,
                    run_id,
                    report.model_dump_json(),
                    report.metrics.tokens.model_dump(),
                )
                await asyncio.to_thread(db.persist_report, report)
                await asyncio.to_thread(db.set_run_status, run_id, report.status.value)
            except asyncio.CancelledError:
                handle.status = RunStatus.FAILED
                handle.error = "cancelled"
                await asyncio.to_thread(db.set_run_status, run_id, "FAILED", "cancelled")
                raise
            except Exception as exc:
                log.error("run task failed", run=run_id, error=str(exc)[:300], exc_info=True)
                handle.status = RunStatus.FAILED
                handle.error = str(exc)[:500]
                await asyncio.to_thread(db.set_run_status, run_id, "FAILED", handle.error)
            finally:
                self._publish(
                    handle,
                    RunEvent(
                        run_id=run_id,
                        node="runner",
                        message="stream complete",
                        payload={"terminal": True, "status": handle.status.value},
                    ),
                )

        handle.task = asyncio.create_task(runner(), name=f"run:{run_id}")
        log.info("run started", run=run_id, topic=topic[:80])
        return handle

    async def cancel(self, run_id: str) -> bool:
        handle = self._runs.get(run_id)
        if handle is None or handle.task is None or handle.finished:
            return False
        handle.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handle.task
        return True

    # ── event fan-out ────────────────────────────────────────────────────────
    def _publish(self, handle: RunHandle, event: RunEvent) -> None:
        handle.events.append(event)
        if len(handle.events) > _MAX_BUFFER:
            dropped = len(handle.events) - _MAX_BUFFER
            del handle.events[:dropped]
            handle.first_seq += dropped

        for queue in list(handle.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A subscriber too slow to keep up is dropped rather than
                # allowed to apply backpressure to the run itself.
                log.debug("dropping slow SSE subscriber", run=handle.run_id)
                handle.subscribers.discard(queue)

        try:
            get_db().append_event(
                handle.run_id, event.node, event.message, event.level, event.payload
            )
        except Exception as exc:
            log.debug("event persist failed", error=str(exc)[:160])

    async def subscribe(self, run_id: str) -> tuple[RunHandle, asyncio.Queue]:
        handle = self._runs.get(run_id)
        if handle is None:
            raise KeyError(run_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        handle.subscribers.add(queue)
        return handle, queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        handle = self._runs.get(run_id)
        if handle is not None:
            handle.subscribers.discard(queue)

    async def stream(self, run_id: str, last_event_id: str | None = None):
        """Yield SSE payloads: buffered history first, then live events.

        Yields dicts, not pre-formatted strings. ``EventSourceResponse`` does the
        wire formatting itself; handing it a raw ``"event: x\\ndata: ...\\n\\n"``
        string gets that string wrapped *again* as a data payload, producing
        frames no client can parse.
        """
        handle, queue = await self.subscribe(run_id)

        # Resume point. A fresh subscriber gets the full history; a reconnecting
        # one gets only what it missed.
        resume_from = -1
        if last_event_id:
            try:
                resume_from = int(last_event_id)
            except ValueError:
                resume_from = -1

        seq = handle.first_seq
        try:
            for event in list(handle.events):
                if seq > resume_from:
                    yield _sse(event, seq)
                seq += 1

            if handle.finished:
                yield _sse(_terminal_event(handle), seq)
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    # Ping keeps proxies from reaping an idle stream.
                    yield {"event": "ping", "data": "{}"}
                    if handle.finished:
                        yield _sse(_terminal_event(handle), seq)
                        return
                    continue

                yield _sse(event, seq)
                seq += 1
                if event.payload.get("terminal"):
                    return
        finally:
            self.unsubscribe(run_id, queue)

    async def load_from_db(self, run_id: str) -> RunHandle | None:
        """Rehydrate a finished run after a process restart."""
        row = await asyncio.to_thread(get_db().get_run, run_id)
        if row is None:
            return None

        handle = RunHandle(
            run_id=run_id,
            topic=str(row["topic"]),
            status=RunStatus(str(row["status"])),
            error=str(row["error"] or ""),
        )
        if row["report_json"]:
            try:
                handle.report = ResearchReport.model_validate_json(str(row["report_json"]))
            except Exception as exc:
                log.warning("stored report unreadable", run=run_id, error=str(exc)[:200])

        events = await asyncio.to_thread(get_db().events_since, run_id, 0, _MAX_BUFFER)
        handle.events = [
            RunEvent(
                run_id=run_id,
                node=str(e["node"]),
                level=str(e["level"]),  # type: ignore[arg-type]
                message=str(e["message"]),
                payload=json.loads(str(e["payload_json"]) or "{}"),
            )
            for e in events
        ]
        self._runs[run_id] = handle
        return handle


def _terminal_event(handle: RunHandle) -> RunEvent:
    return RunEvent(
        run_id=handle.run_id,
        node="runner",
        message="stream complete",
        payload={"terminal": True, "status": handle.status.value},
    )


def _sse(event: RunEvent, seq: int | None = None) -> dict[str, str]:
    """One SSE frame as sse_starlette expects it.

    The ``id`` matters: on a dropped connection EventSource reconnects
    automatically and sends the last id back as ``Last-Event-ID``. Without it
    the server has no way to know what the client already has, replays the
    whole buffer, and the UI shows the entire run a second time.
    """
    payload = json.dumps(
        {
            "ts": event.ts.isoformat(),
            "node": event.node,
            "level": event.level,
            "message": event.message,
            **event.payload,
        },
        default=str,
    )
    frame = {"event": event.node or "message", "data": payload}
    if seq is not None:
        frame["id"] = str(seq)
    return frame


_MANAGER: RunManager | None = None


def get_manager() -> RunManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = RunManager()
    return _MANAGER


def _reset_manager() -> None:
    """Test hook."""
    global _MANAGER
    _MANAGER = None
