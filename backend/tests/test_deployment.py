"""Deployment robustness: database location and container startup.

These cover a failure that reached production. `VOLUME ["/data"]` in the
Dockerfile made Docker materialise a fresh, root-owned anonymous volume at
container start, which shadowed the build-time ``chown``. The unprivileged
runtime user could not create the database, and the app died during startup
with a bare ``sqlite3.OperationalError: unable to open database file`` — no
indication of which path failed or why.

The fix is twofold: drop the VOLUME declaration, and never let an unwritable
configured path be fatal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from veritas.storage.db import Database, resolve_db_path

# A path an unprivileged process cannot write to on any POSIX system.
UNWRITABLE = "/veritas-must-not-be-writable.db"


class TestDatabasePathResolution:
    def test_writable_path_is_used_unchanged(self, tmp_path):
        target = tmp_path / "veritas.db"
        assert resolve_db_path(target) == target

    def test_missing_parent_directories_are_created(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "veritas.db"
        assert resolve_db_path(target) == target
        assert target.parent.is_dir()

    def test_unwritable_location_falls_back_instead_of_raising(self):
        """The exact production failure: startup must survive it."""
        resolved = resolve_db_path(UNWRITABLE)
        assert resolved != Path(UNWRITABLE)
        assert resolved.parent.is_dir()

    def test_fallback_preserves_the_filename(self):
        resolved = resolve_db_path(UNWRITABLE)
        assert resolved.name == Path(UNWRITABLE).name

    def test_fallback_target_is_actually_usable(self):
        """A fallback that is itself unwritable would be worthless.

        `mkdir(exist_ok=True)` succeeds on a directory owned by someone else,
        so the resolver probes with a real file write rather than trusting it.
        """
        resolved = resolve_db_path(UNWRITABLE)
        db = Database(resolved)
        try:
            db.create_run("run_probe", "a topic", {})
            assert db.get_run("run_probe") is not None
        finally:
            db.close()
            resolved.unlink(missing_ok=True)

    def test_write_probe_leaves_nothing_behind(self, tmp_path):
        resolve_db_path(tmp_path / "veritas.db")
        leftovers = list(tmp_path.glob(".veritas-write-test-*"))
        assert not leftovers, f"write probe left files behind: {leftovers}"

    def test_read_only_directory_falls_back(self, tmp_path):
        """Covers a mounted-but-read-only volume, distinct from ownership."""
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)  # r-x: listable, not writable
        try:
            resolved = resolve_db_path(locked / "veritas.db")
            assert resolved.parent != locked
        finally:
            locked.chmod(0o700)


class TestDockerfileContract:
    """Guards on the deployment files themselves.

    Cheap to check, and each encodes a bug that already cost a failed deploy.
    """

    @property
    def _root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def test_dockerfile_declares_no_volume(self):
        """A VOLUME re-mounts root-owned at runtime and breaks the chown."""
        dockerfile = (self._root / "Dockerfile").read_text()
        offending = [
            line
            for line in dockerfile.splitlines()
            if line.strip().upper().startswith("VOLUME")
        ]
        assert not offending, (
            "VOLUME in the Dockerfile shadows the build-time chown with a "
            f"root-owned mount: {offending}"
        )

    def test_dockerfile_binds_the_platform_port(self):
        """Render and most PaaS inject $PORT and require the app to bind it."""
        dockerfile = (self._root / "Dockerfile").read_text()
        assert "${PORT:-8000}" in dockerfile

    def test_dockerfile_runs_unprivileged(self):
        dockerfile = (self._root / "Dockerfile").read_text()
        assert "USER veritas" in dockerfile

    def test_render_blueprint_keeps_secrets_out_of_git(self):
        render = (self._root / "render.yaml").read_text()
        assert "GEMINI_API_KEY" in render
        assert "sync: false" in render, "API keys must not be committed as values"

    def test_render_uses_a_writable_database_path(self):
        render = (self._root / "render.yaml").read_text()
        assert "/tmp/veritas.db" in render


@pytest.mark.parametrize("path", ["relative.db", "./nested/relative.db"])
def test_relative_paths_resolve_without_error(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_db_path(path)
    assert resolved.parent.is_dir()


class TestHealthSurface:
    """/health is the only way to confirm a deployment picked up its config.

    A provider added to the app but omitted here is invisible: you cannot tell
    whether the key was saved, or whether the new code even deployed.
    """

    def test_every_provider_key_is_reported(self):
        from veritas.config import env_summary

        summary = env_summary()
        for provider in ("openai", "anthropic", "gemini", "groq"):
            assert f"{provider}_key_present" in summary, (
                f"{provider} key status missing from /health — a deploy using it "
                "could not be verified"
            )

    def test_no_secret_values_are_exposed(self):
        """Presence booleans only; never the keys themselves."""
        from veritas.config import env_summary

        for key, value in env_summary().items():
            if key.endswith("_key_present"):
                assert isinstance(value, bool)

    def test_render_declares_every_supported_provider_key(self):
        """An undeclared key gives no field in the Render dashboard to fill in."""
        render = (Path(__file__).resolve().parents[2] / "render.yaml").read_text()
        for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            assert key in render, f"{key} missing from render.yaml"


class TestCredentialHygiene:
    """Whitespace in a pasted key produces an opaque 401.

    A real deployment failed this way: the key was correct but the dashboard
    paste carried a trailing newline, and the provider's error said only
    "Invalid API Key" — nothing about whitespace.
    """

    def test_keys_are_stripped(self):
        from veritas.config import Settings

        s = Settings(  # type: ignore[call-arg]
            GROQ_API_KEY="  gsk_abc \n",
            GEMINI_API_KEY="\tAIzaKey ",
            OPENAI_API_KEY=" sk-x\r\n",
            TAVILY_API_KEY=" tvly-y ",
        )
        assert s.groq_api_key == "gsk_abc"
        assert s.gemini_api_key == "AIzaKey"
        assert s.openai_api_key == "sk-x"
        assert s.tavily_api_key == "tvly-y"

    def test_whitespace_only_key_is_treated_as_absent(self):
        from veritas.config import Settings

        s = Settings(GROQ_API_KEY="   ", GEMINI_API_KEY="", OPENAI_API_KEY="",  # type: ignore[call-arg]
                     ANTHROPIC_API_KEY="", VERITAS_LLM_PROVIDER="auto")
        assert s.groq_api_key == ""
        assert s.resolved_provider == "fake"

    def test_searxng_url_is_stripped(self):
        from veritas.config import Settings

        assert Settings(SEARXNG_URL=" http://x:8080 ").searxng_url == "http://x:8080"  # type: ignore[call-arg]


class TestSearchCostControl:
    def test_tavily_defaults_to_the_cheaper_depth(self):
        """'advanced' costs 2 credits per search vs 1 — halving free-tier runs."""
        from veritas.config import Settings

        assert Settings().tavily_search_depth == "basic"  # type: ignore[call-arg]


class TestSearxngService:
    """The deployed search backend.

    DuckDuckGo blocks datacenter IPs, so the keyless fallback that works
    locally returns nothing from Render. SearXNG self-hosted is the fix.
    """

    @property
    def _root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def test_dockerfile_and_entrypoint_exist(self):
        assert (self._root / "searxng" / "Dockerfile").is_file()
        assert (self._root / "searxng" / "entrypoint.sh").is_file()
        assert (self._root / "searxng" / "settings.yml").is_file()

    def test_entrypoint_maps_platform_port_to_granian(self):
        """The image reads GRANIAN_PORT, not the SEARXNG_* names docs imply."""
        script = (self._root / "searxng" / "entrypoint.sh").read_text()
        assert "GRANIAN_PORT" in script
        assert "${PORT:-8080}" in script

    def test_settings_enable_the_json_api(self):
        """Without this SearXNG answers API calls with 403 and search dies."""
        settings = (self._root / "searxng" / "settings.yml").read_text()
        assert "json" in settings
        assert "limiter: false" in settings

    def test_render_declares_the_searxng_service(self):
        render = (self._root / "render.yaml").read_text()
        assert "veritas-searxng" in render
        assert "./searxng/Dockerfile" in render
        assert "/healthz" in render, "confirmed health path on the live image"

    def test_searxng_url_is_wired_from_the_service(self):
        """Hand-pasting the URL is the step people forget."""
        render = (self._root / "render.yaml").read_text()
        assert "fromService" in render


class TestSearchWarmup:
    """Free-tier services sleep; a judge will arrive cold.

    A run fires several searches at once. Against a sleeping instance they all
    time out together and the run degrades to scholarly sources — two per
    question instead of twenty. The app must wake search itself rather than
    depending on someone loading a URL beforehand.

    Note these tests inject `poll_interval` rather than patching
    `asyncio.sleep`. Patching it reaches the real asyncio module and breaks
    pytest-asyncio's own loop management — the suite hung and was SIGKILLed.
    """

    def _online(self, monkeypatch):
        """Opt this test out of the suite-wide network kill-switch.

        warm_searxng honours `offline` — that guard is what stopped a
        90-second polling task leaking from every API test. Tests that
        exercise the wake path have to disable it deliberately.
        """
        from veritas.config import get_settings

        monkeypatch.setattr(get_settings(), "offline", False)

    def _client(self, responder):
        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                return responder(url)

        return lambda **kw: FakeClient()

    async def test_no_url_configured_is_a_no_op(self):
        from veritas.tools.search import warm_searxng

        assert await warm_searxng("") is False

    async def test_returns_true_once_healthz_answers(self, monkeypatch):
        import httpx

        from veritas.tools import search as search_mod

        seen = {}

        class Ok:
            status_code = 200

        def responder(url):
            seen["url"] = url
            return Ok()

        self._online(monkeypatch)
        monkeypatch.setattr(httpx, "AsyncClient", self._client(responder))
        assert await search_mod.warm_searxng("https://x.test", poll_interval=0.01) is True
        assert seen["url"].endswith("/healthz")

    async def test_retries_a_sleeping_instance_then_succeeds(self, monkeypatch):
        import httpx

        from veritas.tools import search as search_mod

        calls = {"n": 0}

        class Ok:
            status_code = 200

        def responder(url):
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("still waking")
            return Ok()

        self._online(monkeypatch)
        monkeypatch.setattr(httpx, "AsyncClient", self._client(responder))
        result = await search_mod.warm_searxng(
            "https://x.test", timeout=30, poll_interval=0.01
        )
        assert result is True
        assert calls["n"] == 3

    async def test_gives_up_within_the_timeout(self, monkeypatch):
        import httpx

        from veritas.tools import search as search_mod

        def responder(url):
            raise httpx.ConnectError("asleep")

        monkeypatch.setattr(httpx, "AsyncClient", self._client(responder))
        result = await search_mod.warm_searxng(
            "https://x.test", timeout=0.15, poll_interval=0.01
        )
        assert result is False

    def test_bare_hostname_is_accepted(self):
        """render.yaml's fromService yields a hostname with no scheme."""
        from veritas.tools.search import _normalise_base_url

        assert _normalise_base_url("veritas-searxng.onrender.com").startswith("https://")

    async def test_offline_mode_refuses_to_warm(self):
        """The guard that stopped a 90s polling task leaking per API test."""
        from veritas.tools.search import warm_searxng

        # conftest sets VERITAS_OFFLINE=true for the whole suite.
        assert await warm_searxng("https://x.test", poll_interval=0.01) is False


class TestUnresolvableSearchHost:
    """A non-existent hostname and a sleeping one need different handling.

    Waiting 75s is right for a booting instance and pure waste for a host that
    does not exist. Seen in production: SEARXNG_URL was left at the Docker
    Compose service name, which resolves only inside Compose, and every query
    burned ~17s on DNS before failing.
    """

    def test_dns_errors_are_recognised(self):
        from veritas.tools.search import _is_unresolvable

        for msg in (
            "[Errno -2] Name or service not known",
            "nodename nor servname provided",
            "Temporary failure in name resolution",
        ):
            assert _is_unresolvable(Exception(msg)), msg

    def test_transient_errors_are_not_mistaken_for_dns(self):
        """A timeout means asleep — that one IS worth waiting for."""
        from veritas.tools.search import _is_unresolvable

        for msg in ("Connection timed out", "502 Bad Gateway", "Read timeout"):
            assert not _is_unresolvable(Exception(msg)), msg

    def test_search_target_is_visible_for_diagnosis(self):
        """A masked dashboard makes a wrong URL undiagnosable.

        Cost a deployment: SEARXNG_URL looked right in the UI but did not
        resolve, and nothing in the app would say what string it actually held.
        """
        from veritas.config import env_summary

        assert "searxng_url" in env_summary()

    def test_health_never_exposes_an_actual_secret(self):
        from veritas.config import env_summary

        summary = env_summary()
        for key in ("groq_api_key", "gemini_api_key", "openai_api_key", "SEARXNG_SECRET"):
            assert key not in summary


class TestSearxngUrlValidation:
    """A wrong value in SEARXNG_URL must fail loudly, not mysteriously.

    In production this variable was set to the SearXNG *secret* rather than the
    URL. The old code prepended https:// to it, producing a host that could not
    resolve, and the only symptom was a bare DNS error that said nothing about
    the real mistake.
    """

    def test_a_pasted_secret_is_rejected(self):
        from veritas.tools.search import _normalise_base_url

        assert _normalise_base_url("veritas-hackathon-secret-9x7k2m") == ""
        assert _normalise_base_url("https://veritas-hackathon-secret-9x7k2m") == ""

    def test_a_compose_service_name_is_rejected(self):
        """`searxng` resolves inside Compose and nowhere else."""
        from veritas.tools.search import _normalise_base_url

        assert _normalise_base_url("searxng") == ""

    def test_real_urls_are_accepted(self):
        from veritas.tools.search import _normalise_base_url

        assert (
            _normalise_base_url("https://veritas-searxng.onrender.com")
            == "https://veritas-searxng.onrender.com"
        )
        assert (
            _normalise_base_url("veritas-searxng.onrender.com")
            == "https://veritas-searxng.onrender.com"
        )
        assert _normalise_base_url("localhost:8080") == "http://localhost:8080"

    def test_trailing_punctuation_is_stripped(self):
        """Copying a URL out of prose picks up the sentence's full stop."""
        from veritas.tools.search import _normalise_base_url

        assert (
            _normalise_base_url("https://veritas-searxng.onrender.com.")
            == "https://veritas-searxng.onrender.com"
        )

    async def test_a_disabled_provider_returns_empty_not_an_error(self):
        """Blanking base_url made search() build "/search" — a protocol-less URL.

        That turned a clear DNS failure into a confusing UnsupportedProtocol
        error, masking the actual misconfiguration.
        """
        from veritas.tools.search import SearxngProvider

        p = SearxngProvider(None, "https://x.test")  # type: ignore[arg-type]
        p._disabled = True
        assert await p.search("anything", limit=3) == []
        assert p.available is False

    async def test_an_unset_url_returns_empty(self):
        from veritas.tools.search import SearxngProvider

        p = SearxngProvider(None, "")  # type: ignore[arg-type]
        assert await p.search("anything", limit=3) == []
