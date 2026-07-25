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
        response = client.get("/")
        assert response.status_code == 200
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
