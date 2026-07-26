"""HTTP API contract tests."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from veritas.api.app import create_app
from veritas.api.manager import _reset_manager


@pytest.fixture
def client():
    _reset_manager()
    with TestClient(create_app()) as test_client:
        yield test_client
    _reset_manager()


class TestSystemRoutes:
    def test_root(self, client):
        """Serves the exported UI when one is bundled, else service metadata.

        Both are valid: the deployed image ships the static build, while a
        backend-only checkout has nothing to serve.
        """
        response = client.get("/")
        assert response.status_code == 200

        if "html" in response.headers.get("content-type", "").lower():
            assert "<html" in response.text.lower()
        else:
            assert response.json()["name"] == "VERITAS"

    def test_health_exposes_the_active_provider(self, client):
        """A demo must never silently pass offline heuristics off as model output."""
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["config"]["llm_provider"] == "fake"

    def test_openapi_schema_builds(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestRunRoutes:
    def test_create_run_is_accepted_immediately(self, client):
        response = client.post("/api/runs", json={"topic": "renewable energy adoption"})
        assert response.status_code == 202
        body = response.json()
        assert body["run_id"].startswith("run_")
        assert body["stream_url"].endswith("/stream")

    def test_rejects_blank_topic(self, client):
        assert client.post("/api/runs", json={"topic": "  "}).status_code == 422

    def test_rejects_missing_topic(self, client):
        assert client.post("/api/runs", json={}).status_code == 422

    def test_unknown_run_returns_404(self, client):
        assert client.get("/api/runs/run_missing").status_code == 404
        assert client.get("/api/runs/run_missing/report").status_code == 404
        assert client.get("/api/runs/run_missing/graph").status_code == 404

    def test_run_completes_and_exposes_report_and_graph(self, client):
        run_id = client.post("/api/runs", json={"topic": "solar power costs"}).json()["run_id"]

        # The run executes on the app's event loop; poll until it settles.
        deadline = time.time() + 90
        body = {}
        while time.time() < deadline:
            body = client.get(f"/api/runs/{run_id}").json()
            if body["finished"]:
                break
            time.sleep(0.4)

        assert body.get("finished"), f"run did not finish: {body.get('status')}"
        assert body["status"] in {"COMPLETED", "BUDGET_EXCEEDED"}
        assert body["event_count"] > 0

        graph = client.get(f"/api/runs/{run_id}/graph")
        assert graph.status_code == 200
        payload = graph.json()
        assert "nodes" in payload and "edges" in payload
        assert isinstance(payload["metrics"]["total_claims"], int)

        assert client.get(f"/api/runs/{run_id}/report").status_code == 200

    def test_list_runs(self, client):
        client.post("/api/runs", json={"topic": "a listed topic"})
        response = client.get("/api/runs")
        assert response.status_code == 200
        assert any(r["topic"] == "a listed topic" for r in response.json())


class TestConfidenceExplainer:
    def test_returns_a_full_breakdown(self, client):
        response = client.get(
            "/api/confidence/explain",
            params={"entail_max": 0.9, "agreement": 1.0, "n_independent": 1},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["contributions"]) == 7
        assert 0.0 <= body["final_score"] <= 1.0

    def test_single_source_is_capped(self, client):
        """The independence ceiling must be visible through the API."""
        body = client.get(
            "/api/confidence/explain",
            params={
                "entail_max": 1.0,
                "agreement": 1.0,
                "independence": 1.0,
                "source_quality": 1.0,
                "consistency": 1.0,
                "sufficiency": 1.0,
                "stated_conf": 1.0,
                "n_independent": 1,
            },
        ).json()

        assert body["capped"] is True
        assert body["final_score"] == body["independence_ceiling"]
        assert body["final_score"] < body["calibrated_score"]

    def test_many_sources_lifts_the_cap(self, client):
        params = {
            "entail_max": 1.0,
            "agreement": 1.0,
            "independence": 1.0,
            "source_quality": 1.0,
            "consistency": 1.0,
            "sufficiency": 1.0,
            "stated_conf": 1.0,
        }
        one = client.get("/api/confidence/explain", params={**params, "n_independent": 1}).json()
        many = client.get("/api/confidence/explain", params={**params, "n_independent": 8}).json()
        assert many["final_score"] > one["final_score"]


class TestVerifyRoute:
    def test_single_claim_verification(self, client):
        response = client.post(
            "/api/verify", json={"claim": "Water boils at 100 degrees Celsius at sea level."}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["claim"]["verdict"] in {"SUPPORTED", "REFUTED", "NEI"}
        assert 0.0 <= body["claim"]["confidence"] <= 1.0

    def test_rejects_too_short_claim(self, client):
        assert client.post("/api/verify", json={"claim": "a"}).status_code == 422


class TestCacheRoutes:
    def test_stats_and_clear(self, client):
        assert client.get("/api/cache/stats").status_code == 200
        assert client.delete("/api/cache").status_code == 200


class TestStreamFormat:
    """Regression guard for the SSE wire format.

    `EventSourceResponse` formats frames itself. Yielding a pre-formatted
    "event: x\\ndata: y\\n\\n" string gets it wrapped a second time, producing
    frames no EventSource can parse — and the failure is silent: the HTTP
    request succeeds, the browser just reconnects forever with no events.
    """

    def test_frames_are_dicts_not_strings(self):
        from veritas.api.manager import _sse
        from veritas.schemas import RunEvent

        frame = _sse(
            RunEvent(run_id="r", node="verifier", message="hi", payload={"confidence": 0.8})
        )
        assert isinstance(frame, dict)
        assert frame["event"] == "verifier"
        assert "\ndata:" not in frame["data"]

    def test_data_is_valid_json_with_payload_merged(self):
        import json as json_mod

        from veritas.api.manager import _sse
        from veritas.schemas import RunEvent

        frame = _sse(
            RunEvent(run_id="r", node="runner", message="done", payload={"terminal": True})
        )
        decoded = json_mod.loads(frame["data"])
        assert decoded["node"] == "runner"
        assert decoded["terminal"] is True

    def test_nodeless_event_falls_back_to_message_channel(self):
        from veritas.api.manager import _sse
        from veritas.schemas import RunEvent

        assert _sse(RunEvent(run_id="r", node="", message="x"))["event"] == "message"

    def test_stream_endpoint_emits_parseable_frames(self, client):
        import json as json_mod

        run_id = client.post("/api/runs", json={"topic": "sse format check"}).json()["run_id"]

        deadline = time.time() + 90
        while time.time() < deadline:
            if client.get(f"/api/runs/{run_id}").json()["finished"]:
                break
            time.sleep(0.4)

        with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        data_lines = [
            line[len("data: ") :]
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert data_lines, "stream produced no data frames"
        for line in data_lines:
            decoded = json_mod.loads(line)  # would raise on a double-wrapped frame
            assert isinstance(decoded, dict)


class TestStaticFrontendRouting:
    """Single-origin serving: FastAPI hosts the exported UI and the API.

    The failure this guards against actually shipped: the SPA catch-all raised
    an exception class that was never imported, so every unmatched /api/* path
    returned 500 instead of 404. `create_app()` still succeeded, because a
    NameError inside a function body only fires when that function runs.
    """

    def test_unknown_api_path_is_404_not_the_html_shell(self, client):
        response = client.get("/api/definitely-not-a-real-endpoint")
        assert response.status_code == 404
        assert "html" not in response.headers.get("content-type", "").lower()

    def test_real_api_routes_still_win_over_the_catch_all(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200

    def test_root_responds(self, client):
        """Serves the UI when bundled, service metadata when not."""
        assert client.get("/").status_code == 200

    def test_frontend_dir_requires_an_index(self, tmp_path, monkeypatch):
        from veritas.api.app import _frontend_dir

        monkeypatch.setenv("VERITAS_STATIC_DIR", str(tmp_path))
        assert _frontend_dir() != tmp_path, "a directory with no index.html is not a build"

        (tmp_path / "index.html").write_text("<html></html>")
        assert _frontend_dir() == tmp_path

    def test_traversal_outside_the_export_is_refused(self, tmp_path, monkeypatch):
        """A path escaping the export dir must not serve arbitrary files."""
        static = tmp_path / "static"
        static.mkdir()
        (static / "index.html").write_text("<html>ui</html>")
        secret = tmp_path / "secret.txt"
        secret.write_text("do-not-serve")

        monkeypatch.setenv("VERITAS_STATIC_DIR", str(static))
        _reset_manager()
        with TestClient(create_app()) as c:
            body = c.get("/../secret.txt").text
        assert "do-not-serve" not in body


class TestPayloadSize:
    """A finished run's payload must not carry server-side-only bulk.

    `Source.content` holds up to 60,000 chars of extracted page text per source
    — needed by the verification pipeline, never read by the browser. Shipping
    it made a real run's response ~538 KB (82% dead weight), slow enough on a
    small instance that the browser aborted with "Load failed" after an
    otherwise successful 5-minute run.
    """

    def test_source_content_is_stripped_from_the_wire(self):
        from veritas.api.routes import public_report
        from veritas.schemas import ResearchReport, Source

        report = ResearchReport(
            run_id="r",
            topic="t",
            sources=[Source(url="https://x.test", content="X" * 50_000, title="T")],
        )
        payload = public_report(report)
        assert "content" not in payload["sources"][0]

    def test_fields_the_ui_needs_survive(self):
        from veritas.api.routes import public_report
        from veritas.schemas import ResearchReport, Source

        report = ResearchReport(
            run_id="r",
            topic="t",
            sources=[
                Source(url="https://x.test", domain="x.test", title="Title", content="X" * 9_000)
            ],
        )
        src = public_report(report)["sources"][0]
        for field in ("id", "url", "domain", "title", "credibility_tier", "credibility_score"):
            assert field in src, f"UI needs {field}"

    def test_payload_shrinks_substantially(self):
        import json as json_mod

        from veritas.api.routes import public_report
        from veritas.schemas import ResearchReport, Source

        report = ResearchReport(
            run_id="r",
            topic="t",
            sources=[
                Source(url=f"https://x{i}.test", content="X" * 45_000) for i in range(10)
            ],
        )
        full = len(json_mod.dumps(report.model_dump(mode="json")))
        lean = len(json_mod.dumps(public_report(report)))
        assert lean < full * 0.3, f"expected a large reduction, got {lean} vs {full}"

    def test_endpoint_response_excludes_content(self, client):
        import time as _time

        run_id = client.post("/api/runs", json={"topic": "payload size check"}).json()["run_id"]
        deadline = _time.time() + 90
        while _time.time() < deadline:
            body = client.get(f"/api/runs/{run_id}").json()
            if body["finished"]:
                break
            _time.sleep(0.4)

        report = body.get("report")
        if report and report.get("sources"):
            assert all("content" not in s for s in report["sources"])


class TestStreamResume:
    """A dropped connection must not replay the whole run.

    EventSource reconnects automatically. Without an `id` on each frame the
    server has no way to know what the client already received, so it replayed
    the entire buffer — and the UI showed the run twice end to end. Observed on
    an 11-minute production run.
    """

    def _handle(self):
        from veritas.api.manager import RunHandle
        from veritas.schemas import RunEvent, RunStatus

        # COMPLETED so the stream terminates after replay instead of blocking
        # on the live queue — we are testing the replay path only.
        h = RunHandle(run_id="r", topic="t", status=RunStatus.COMPLETED)
        h.events = [RunEvent(run_id="r", node="verifier", message=f"e{i}") for i in range(5)]
        return h

    def test_frames_carry_a_sequential_id(self):
        from veritas.api.manager import _sse
        from veritas.schemas import RunEvent

        frame = _sse(RunEvent(run_id="r", node="planner", message="x"), seq=7)
        assert frame["id"] == "7"

    def test_id_is_omitted_when_not_supplied(self):
        from veritas.api.manager import _sse
        from veritas.schemas import RunEvent

        assert "id" not in _sse(RunEvent(run_id="r", node="planner", message="x"))

    async def test_fresh_subscriber_gets_full_history(self):
        from veritas.api.manager import RunManager

        m = RunManager()
        m._runs["r"] = self._handle()
        frames = [f async for f in m.stream("r", last_event_id=None)]
        assert len([f for f in frames if f.get("event") == "verifier"]) == 5

    async def test_reconnecting_subscriber_gets_only_the_remainder(self):
        """The exact bug: without this the client sees every event twice."""
        from veritas.api.manager import RunManager

        m = RunManager()
        m._runs["r"] = self._handle()
        frames = [f async for f in m.stream("r", last_event_id="2")]
        replayed = [f for f in frames if f.get("event") == "verifier"]
        assert len(replayed) == 2, "should resume after id 2, not replay all 5"
        assert [f["id"] for f in replayed] == ["3", "4"]

    async def test_malformed_last_event_id_falls_back_to_full_replay(self):
        from veritas.api.manager import RunManager

        m = RunManager()
        m._runs["r"] = self._handle()
        frames = [f async for f in m.stream("r", last_event_id="not-a-number")]
        assert len([f for f in frames if f.get("event") == "verifier"]) == 5

    def test_trimming_the_buffer_does_not_renumber_events(self):
        """A list index is not a stable id once the buffer is trimmed."""
        from veritas.api.manager import _MAX_BUFFER, RunHandle, RunManager
        from veritas.schemas import RunEvent, RunStatus

        m = RunManager()
        h = RunHandle(run_id="r", topic="t", status=RunStatus.RUNNING)
        m._runs["r"] = h
        for i in range(_MAX_BUFFER + 25):
            m._publish(h, RunEvent(run_id="r", node="verifier", message=f"e{i}"))

        assert h.first_seq == 25, "sequence floor must advance as events are dropped"
        assert len(h.events) == _MAX_BUFFER


class TestSearchHealthCaching:
    """The probe runs a live search; on a small instance it measured 15.1s.

    Fired on every page load, competing for the same 0.1 CPU the run needs.
    Retrieval health does not change second to second, so a slightly stale
    answer is worth far more than a fresh one that slows the app down.
    """

    def test_second_call_is_served_from_cache(self, client):
        first = client.get("/api/search/health").json()
        second = client.get("/api/search/health").json()
        assert first.get("cached") is False
        assert second.get("cached") is True
        assert second["working"] == first["working"]

    def test_fresh_bypasses_the_cache(self, client):
        client.get("/api/search/health")
        forced = client.get("/api/search/health", params={"fresh": True}).json()
        assert forced.get("cached") is False

    def test_cached_response_reports_its_age(self, client):
        client.get("/api/search/health")
        second = client.get("/api/search/health").json()
        assert "age_seconds" in second
