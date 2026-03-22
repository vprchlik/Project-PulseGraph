const API_BASE = "/api";

export interface RegimeProbs {
  breakout: number;
  steady_growth: number;
  plateau: number;
  spike_and_fade: number;
  decay: number;
}

export interface ForecastData {
  repo_name: string;
  horizon: number;
  model_name: string;
  regime_probs: RegimeProbs;
  dominant_regime: string;
  dominant_prob: number;
  median_forecast: number[];
  lower_80: number[];
  upper_80: number[];
  lower_95: number[];
  upper_95: number[];
  confidence_score: number;
  context_length: number;
}

export interface RegimeSearchResult {
  repo_name: string;
  regime_prob: number;
  dominant_regime: string;
  stars_current: number;
  language: string;
  description: string;
}

export interface RegimeSearchResponse {
  regime: string;
  horizon: number;
  min_prob: number;
  results: RegimeSearchResult[];
  total_count: number;
}

export interface AnalogMatch {
  match_repo: string;
  match_date: string;
  similarity: number;
  realized_outcome: string;
}

export interface EventSummary {
  event_type: string;
  event_date: string;
  title: string;
  url: string;
  source: string;
  relevance_score: number;
}

export interface HealthData {
  status: string;
  version: string;
  repos_loaded: number;
  model_loaded: boolean;
}

export async function fetchForecast(
  owner: string,
  repo: string,
  horizon: number = 30
): Promise<ForecastData> {
  const res = await fetch(
    `${API_BASE}/entity/${owner}/${repo}/forecast?horizon=${horizon}`
  );
  if (!res.ok) throw new Error(`Forecast failed: ${res.statusText}`);
  return res.json();
}

export async function searchRegime(
  regime: string,
  horizon: number = 30,
  minProb: number = 0.3,
  limit: number = 50
): Promise<RegimeSearchResponse> {
  const params = new URLSearchParams({
    regime,
    horizon: String(horizon),
    min_prob: String(minProb),
    limit: String(limit),
  });
  const res = await fetch(`${API_BASE}/search/regime?${params}`);
  if (!res.ok) throw new Error(`Regime search failed: ${res.statusText}`);
  return res.json();
}

export async function searchAnalogs(
  owner: string,
  repo: string,
  lookback: number = 30,
  k: number = 10
): Promise<{ query_repo: string; matches: AnalogMatch[] }> {
  const params = new URLSearchParams({
    owner,
    repo,
    lookback: String(lookback),
    k: String(k),
  });
  const res = await fetch(`${API_BASE}/search/analog?${params}`);
  if (!res.ok) throw new Error(`Analog search failed: ${res.statusText}`);
  return res.json();
}

export async function fetchExplanation(
  owner: string,
  repo: string
): Promise<{ repo_name: string; events: EventSummary[]; narrative: string }> {
  const res = await fetch(`${API_BASE}/entity/${owner}/${repo}/explain`);
  if (!res.ok) throw new Error(`Explain failed: ${res.statusText}`);
  return res.json();
}

export async function fetchHealth(): Promise<HealthData> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}
