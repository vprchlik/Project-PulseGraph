"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegimeProbs(BaseModel):
    breakout: float = 0.0
    steady_growth: float = 0.0
    plateau: float = 0.0
    spike_and_fade: float = 0.0
    decay: float = 0.0


class ForecastResponse(BaseModel):
    repo_name: str
    horizon: int
    model_name: str
    regime_probs: RegimeProbs
    dominant_regime: str
    dominant_prob: float
    median_forecast: list[float]
    lower_80: list[float]
    upper_80: list[float]
    lower_95: list[float]
    upper_95: list[float]
    confidence_score: float = 0.0
    context_length: int = 0


class EventSummary(BaseModel):
    event_type: str
    event_date: str
    title: str
    url: str = ""
    source: str = ""
    relevance_score: float = 0.0


class ExplanationResponse(BaseModel):
    repo_name: str
    events: list[EventSummary]
    narrative: str = ""
    confidence_score: float = 0.0


class AnalogMatchResponse(BaseModel):
    match_repo: str
    match_date: str
    similarity: float
    realized_outcome: str = ""


class AnalogSearchResponse(BaseModel):
    query_repo: str
    matches: list[AnalogMatchResponse]


class RegimeSearchResult(BaseModel):
    repo_name: str
    regime_prob: float
    dominant_regime: str
    stars_current: int = 0
    language: str = ""
    description: str = ""


class RegimeSearchResponse(BaseModel):
    regime: str
    horizon: int
    min_prob: float
    results: list[RegimeSearchResult]
    total_count: int


class CalibrationInfo(BaseModel):
    repo_name: str
    confidence_score: float
    signal_history_days: int
    signal_volatility: float
    data_completeness: float
    coverage_80: float | None = None
    coverage_95: float | None = None
    abstain: bool = False
    abstain_reason: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    repos_loaded: int = 0
    model_loaded: bool = False
