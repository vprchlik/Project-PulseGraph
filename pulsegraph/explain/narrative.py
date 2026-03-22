"""LLM-based narrative generation for forecast explanations.

Produces human-readable summaries of forecast results, regime analyses,
and event attributions using a lightweight LLM call.
"""

from __future__ import annotations

import logging

import httpx

from pulsegraph.config import settings

logger = logging.getLogger(__name__)


def generate_narrative(
    repo_name: str,
    regime_probs: dict[str, float],
    dominant_regime: str,
    horizon: int,
    events: list[dict] | None = None,
    analog_matches: list[dict] | None = None,
) -> str:
    """Generate a narrative summary of the forecast using an LLM.

    Falls back to a template-based summary if the LLM is unavailable.
    """
    if not settings.llm_api_key:
        return _template_narrative(
            repo_name, regime_probs, dominant_regime, horizon, events
        )

    prompt = _build_prompt(
        repo_name, regime_probs, dominant_regime, horizon, events, analog_matches
    )

    try:
        return _call_llm(prompt)
    except Exception as e:
        logger.warning("LLM call failed, using template: %s", e)
        return _template_narrative(
            repo_name, regime_probs, dominant_regime, horizon, events
        )


def _build_prompt(
    repo_name: str,
    regime_probs: dict[str, float],
    dominant_regime: str,
    horizon: int,
    events: list[dict] | None,
    analog_matches: list[dict] | None,
) -> str:
    probs_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(regime_probs.items(), key=lambda x: -x[1]))
    prompt = f"""Summarize this forecast in 2-3 sentences for a non-technical reader.

Repository: {repo_name}
Forecast horizon: {horizon} days
Dominant regime: {dominant_regime}
Regime probabilities: {probs_str}
"""

    if events:
        top_events = events[:3]
        events_str = "; ".join(
            f"{e.get('title', 'Unknown')} ({e.get('event_date', '')})"
            for e in top_events
        )
        prompt += f"\nRecent events: {events_str}"

    if analog_matches:
        analogs_str = "; ".join(
            f"{m.get('match_repo', '')} ({m.get('match_date', '')})"
            for m in analog_matches[:3]
        )
        prompt += f"\nHistorical analogs: {analogs_str}"

    prompt += "\n\nWrite a concise, factual summary. Do not speculate beyond what the data shows. Frame regime probabilities as possibilities, not certainties."

    return prompt


def _call_llm(prompt: str) -> str:
    """Call the configured LLM API."""
    if settings.llm_provider == "openai":
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a concise financial/tech analyst."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    elif settings.llm_provider == "anthropic":
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.llm_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def _template_narrative(
    repo_name: str,
    regime_probs: dict[str, float],
    dominant_regime: str,
    horizon: int,
    events: list[dict] | None = None,
) -> str:
    """Simple template-based narrative when LLM is unavailable."""
    prob = regime_probs.get(dominant_regime, 0.0)

    narrative = (
        f"Over the next {horizon} days, {repo_name} is most likely to follow "
        f"a {dominant_regime.replace('_', ' ')} trajectory ({prob:.0%} probability)."
    )

    sorted_regimes = sorted(regime_probs.items(), key=lambda x: -x[1])
    if len(sorted_regimes) > 1 and sorted_regimes[1][1] > 0.2:
        alt = sorted_regimes[1]
        narrative += (
            f" An alternative scenario is {alt[0].replace('_', ' ')} "
            f"({alt[1]:.0%} probability)."
        )

    if events:
        top_event = events[0]
        narrative += (
            f" The most relevant recent event is \"{top_event.get('title', 'unknown')}\" "
            f"({top_event.get('event_date', '')})."
        )

    return narrative
