"""Central configuration for PulseGraph."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = ROOT_DIR / "reports"

for d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    github_token: str = ""
    bigquery_project_id: str = ""
    google_application_credentials: str = ""

    database_url: str = "postgresql://pulsegraph:pulsegraph@localhost:5432/pulsegraph"

    llm_api_key: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    chronos_model_name: str = "amazon/chronos-bolt-base"
    forecast_horizons: list[int] = Field(default_factory=lambda: [7, 30, 90])
    trajectory_samples: int = 200

    target_repo_count_high: int = 10_000
    target_repo_count_tail: int = 5_000
    tail_star_min: int = 100
    tail_star_max: int = 1_000

    model_config = {"env_file": str(ROOT_DIR / ".env"), "env_file_encoding": "utf-8"}


settings = Settings()
