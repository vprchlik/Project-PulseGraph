"use client";

import { useEffect, useState } from "react";
import {
  fetchForecast,
  fetchExplanation,
  searchAnalogs,
  type ForecastData,
  type EventSummary,
  type AnalogMatch,
} from "@/lib/api";
import { ForecastChart } from "./ForecastChart";
import { RegimeBar } from "./RegimeBar";
import { ArrowLeft, ExternalLink } from "lucide-react";

interface Props {
  repoName: string;
  onBack: () => void;
}

export function EntityDetail({ repoName, onBack }: Props) {
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [analogs, setAnalogs] = useState<AnalogMatch[]>([]);
  const [horizon, setHorizon] = useState(30);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [owner, repo] = repoName.split("/");

  useEffect(() => {
    setLoading(true);
    setError("");

    Promise.all([
      fetchForecast(owner, repo, horizon),
      fetchExplanation(owner, repo),
      searchAnalogs(owner, repo),
    ])
      .then(([fc, ex, an]) => {
        setForecast(fc);
        setEvents(ex.events);
        setAnalogs(an.matches);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [owner, repo, horizon]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-pulse text-gray-400">Loading forecast...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-20">
        <p className="text-red-500 mb-4">{error}</p>
        <button onClick={onBack} className="text-pulse-600 hover:underline">Go back</button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">{repoName}</h2>
          <a
            href={`https://github.com/${repoName}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-pulse-600 hover:underline flex items-center gap-1"
          >
            View on GitHub <ExternalLink size={12} />
          </a>
        </div>

        <div className="ml-auto flex gap-2">
          {[7, 30, 90].map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                horizon === h
                  ? "bg-pulse-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {h}d
            </button>
          ))}
        </div>
      </div>

      {forecast && (
        <>
          <ForecastChart forecast={forecast} />
          <RegimeBar
            regimeProbs={forecast.regime_probs}
            dominant={forecast.dominant_regime}
          />
        </>
      )}

      {events.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
            Correlated Events
          </h3>
          <div className="space-y-3">
            {events.slice(0, 8).map((ev, i) => (
              <div key={i} className="flex items-start gap-3 text-sm">
                <span className="shrink-0 px-2 py-0.5 rounded bg-gray-100 text-gray-500 text-xs font-medium">
                  {ev.event_type}
                </span>
                <div className="min-w-0">
                  <p className="font-medium text-gray-800 truncate">{ev.title}</p>
                  <p className="text-gray-400 text-xs">
                    {ev.event_date} &middot; {ev.source}
                    {ev.relevance_score > 0 &&
                      ` · relevance: ${(ev.relevance_score * 100).toFixed(0)}%`}
                  </p>
                </div>
                {ev.url && (
                  <a href={ev.url} target="_blank" rel="noopener noreferrer" className="shrink-0 text-pulse-500">
                    <ExternalLink size={14} />
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {analogs.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
            Historical Analogs
          </h3>
          <div className="space-y-2">
            {analogs.slice(0, 5).map((a, i) => (
              <div key={i} className="flex items-center justify-between text-sm py-2 border-b border-gray-100 last:border-0">
                <span className="font-medium text-gray-800">{a.match_repo}</span>
                <span className="text-gray-400 text-xs">{a.match_date}</span>
                <span className="tabular-nums text-pulse-600 font-medium">
                  {(a.similarity * 100).toFixed(0)}% similar
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
