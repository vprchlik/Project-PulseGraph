"""API route definitions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from pulsegraph import __version__
from pulsegraph.api.models import (
    AnalogMatchResponse,
    AnalogSearchResponse,
    CalibrationInfo,
    EventSummary,
    ExplanationResponse,
    ForecastResponse,
    HealthResponse,
    RegimeProbs,
    RegimeSearchResponse,
    RegimeSearchResult,
)
from pulsegraph.regime.classifier import Regime

router = APIRouter()


def _get_state(request: Request):
    return request.app.state.pg


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    state = _get_state(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        repos_loaded=len(state.dataset),
        model_loaded=state.model_loaded,
    )


@router.get("/entity/{owner}/{repo}/forecast", response_model=ForecastResponse)
def entity_forecast(
    owner: str,
    repo: str,
    request: Request,
    horizon: int = Query(default=30, ge=1, le=365),
):
    state = _get_state(request)
    repo_name = f"{owner}/{repo}"

    result = state.get_forecast(repo_name, horizon)
    if result is None:
        raise HTTPException(404, f"Repository {repo_name} not found or insufficient data")

    forecast = result["forecast"]
    regime = result["regime"]

    probs = {r.value: regime.regime_probs.get(r, 0.0) for r in Regime}

    return ForecastResponse(
        repo_name=repo_name,
        horizon=horizon,
        model_name=forecast.model_name,
        regime_probs=RegimeProbs(**probs),
        dominant_regime=regime.dominant_regime.value,
        dominant_prob=regime.dominant_prob,
        median_forecast=forecast.median.tolist(),
        lower_80=forecast.lower_80.tolist(),
        upper_80=forecast.upper_80.tolist(),
        lower_95=forecast.lower_95.tolist(),
        upper_95=forecast.upper_95.tolist(),
        context_length=forecast.context_length,
    )


@router.get("/entity/{owner}/{repo}/explain", response_model=ExplanationResponse)
def entity_explain(owner: str, repo: str, request: Request):
    state = _get_state(request)
    repo_name = f"{owner}/{repo}"

    if repo_name not in state.dataset:
        raise HTTPException(404, f"Repository {repo_name} not found")

    from pulsegraph.explain.attribution import attribute_events

    series = state.dataset[repo_name]
    events = attribute_events(repo_name, series)

    event_summaries = [
        EventSummary(
            event_type=e.get("event_type", ""),
            event_date=str(e.get("event_date", "")),
            title=e.get("title", ""),
            url=e.get("url", ""),
            source=e.get("source", ""),
            relevance_score=e.get("relevance_score", 0.0),
        )
        for e in events
    ]

    return ExplanationResponse(
        repo_name=repo_name,
        events=event_summaries[:10],
    )


@router.get("/search/regime", response_model=RegimeSearchResponse)
def search_regime(
    request: Request,
    regime: str = Query(default="breakout"),
    horizon: int = Query(default=30, ge=1, le=365),
    min_prob: float = Query(default=0.3, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=500),
):
    state = _get_state(request)

    try:
        target_regime = Regime(regime)
    except ValueError:
        raise HTTPException(400, f"Invalid regime: {regime}. Options: {[r.value for r in Regime]}")

    results = []
    for repo_name in state.dataset:
        forecast_result = state.get_forecast(repo_name, horizon)
        if forecast_result is None:
            continue

        regime_analysis = forecast_result["regime"]
        prob = regime_analysis.regime_probs.get(target_regime, 0.0)

        if prob >= min_prob:
            meta = state.repo_metadata.get(repo_name, {})
            results.append(RegimeSearchResult(
                repo_name=repo_name,
                regime_prob=prob,
                dominant_regime=regime_analysis.dominant_regime.value,
                stars_current=meta.get("total_stars", 0),
                language=meta.get("language", ""),
                description=meta.get("description", ""),
            ))

    results.sort(key=lambda r: r.regime_prob, reverse=True)
    results = results[:limit]

    return RegimeSearchResponse(
        regime=regime,
        horizon=horizon,
        min_prob=min_prob,
        results=results,
        total_count=len(results),
    )


@router.get("/search/analog", response_model=AnalogSearchResponse)
def search_analog(
    request: Request,
    owner: str = Query(...),
    repo: str = Query(...),
    lookback: int = Query(default=30, ge=7, le=365),
    k: int = Query(default=10, ge=1, le=50),
):
    state = _get_state(request)
    repo_name = f"{owner}/{repo}"

    if repo_name not in state.dataset:
        raise HTTPException(404, f"Repository {repo_name} not found")

    series = state.dataset[repo_name]
    matches = state.analog_index.query_from_series(repo_name, series, window_size=lookback, k=k)

    return AnalogSearchResponse(
        query_repo=repo_name,
        matches=[
            AnalogMatchResponse(
                match_repo=m.match_repo,
                match_date=m.match_date,
                similarity=m.similarity,
                realized_outcome=m.realized_outcome,
            )
            for m in matches
        ],
    )


@router.get("/entity/{owner}/{repo}/calibration", response_model=CalibrationInfo)
def entity_calibration(owner: str, repo: str, request: Request):
    state = _get_state(request)
    repo_name = f"{owner}/{repo}"

    if repo_name not in state.dataset:
        raise HTTPException(404, f"Repository {repo_name} not found")

    from pulsegraph.explain.confidence import compute_confidence

    series = state.dataset[repo_name]
    confidence = compute_confidence(repo_name, series)

    return CalibrationInfo(**confidence)
