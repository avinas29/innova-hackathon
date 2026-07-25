"""Shared fixtures.

Every test runs against the deterministic offline provider with search
disabled, so the suite is hermetic: no API keys, no network, no flakes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Must be set before veritas.config is first imported.
os.environ.update(
    {
        "VERITAS_LLM_PROVIDER": "fake",
        "VERITAS_SEARCH_ORDER": "",
        # Hard network kill-switch: the suite must never make a live call.
        "VERITAS_OFFLINE": "true",
        "VERITAS_CACHE_ENABLED": "false",
        "VERITAS_LOG_LEVEL": "WARNING",
        "VERITAS_MAX_RESEARCH_QUESTIONS": "2",
        "VERITAS_MAX_CLAIMS": "6",
        "VERITAS_CONSISTENCY_SAMPLES": "1",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "TAVILY_API_KEY": "",
        "EXA_API_KEY": "",
        "BRAVE_API_KEY": "",
    }
)


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """Give each test its own SQLite file."""
    from veritas.config import reset_settings_cache
    from veritas.storage.db import reset_db

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        monkeypatch.setenv("VERITAS_DB_PATH", str(db_path))
        reset_settings_cache()
        reset_db()
        yield db_path
        reset_db()
        reset_settings_cache()


@pytest.fixture
def settings():
    from veritas.config import get_settings

    return get_settings()


@pytest.fixture
def offline_llm():
    from veritas.config import get_settings
    from veritas.llm.client import LLMClient, OfflineProvider

    return LLMClient(get_settings(), provider=OfflineProvider())


@pytest.fixture
def sample_evidence():
    from veritas.schemas import Evidence, Stance

    return [
        Evidence(
            claim_id="clm_1",
            source_id="src_1",
            url="https://nature.com/article-1",
            domain="nature.com",
            snippet="Global sea level rose by 21 centimetres between 1900 and 2018 according to "
            "the assessment.",
            stance=Stance.SUPPORTS,
            entailment_score=0.9,
            relevance=0.8,
            credibility_score=1.0,
        ),
        Evidence(
            claim_id="clm_1",
            source_id="src_2",
            url="https://reuters.com/article-2",
            domain="reuters.com",
            snippet="Sea levels have risen roughly 21 cm since 1900, the report found.",
            stance=Stance.SUPPORTS,
            entailment_score=0.85,
            relevance=0.75,
            credibility_score=0.85,
        ),
        Evidence(
            claim_id="clm_1",
            source_id="src_3",
            url="https://contentfarm.example/repost",
            domain="contentfarm.example",
            snippet="Global sea level rose by 21 centimetres between 1900 and 2018 according to "
            "the assessment.",
            stance=Stance.SUPPORTS,
            entailment_score=0.7,
            relevance=0.6,
            credibility_score=0.1,
        ),
    ]
