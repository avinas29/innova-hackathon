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
