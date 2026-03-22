"use client";

import { useEffect, useState } from "react";
import { fetchHealth, type HealthData } from "@/lib/api";
import { SearchPanel } from "@/components/SearchPanel";
import { EntityDetail } from "@/components/EntityDetail";
import { Activity } from "lucide-react";

export default function Home() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Activity className="text-pulse-600" size={28} />
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
            PulseGraph
          </h1>
        </div>
        <p className="text-gray-500 text-lg">
          Search over plausible futures for open-source repositories
        </p>
        {health && (
          <div className="mt-3 flex gap-4 text-xs text-gray-400">
            <span>
              Status:{" "}
              <span className={health.status === "ok" ? "text-emerald-500" : "text-red-500"}>
                {health.status}
              </span>
            </span>
            <span>{health.repos_loaded} repos loaded</span>
            <span>Model: {health.model_loaded ? "Chronos-2" : "ETS fallback"}</span>
            <span>v{health.version}</span>
          </div>
        )}
      </header>

      {selectedRepo ? (
        <EntityDetail
          repoName={selectedRepo}
          onBack={() => setSelectedRepo(null)}
        />
      ) : (
        <div className="space-y-6">
          <SearchPanel onSelectRepo={setSelectedRepo} />

          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
              Quick Start
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
              <div className="rounded-lg border border-gray-100 p-4">
                <h4 className="font-semibold text-gray-800 mb-1">1. Search regimes</h4>
                <p className="text-gray-500">
                  Find repos likely to break out, plateau, or decay over the
                  next 7-90 days using the regime search above.
                </p>
              </div>
              <div className="rounded-lg border border-gray-100 p-4">
                <h4 className="font-semibold text-gray-800 mb-1">2. Explore forecasts</h4>
                <p className="text-gray-500">
                  Click any result to see the full probabilistic forecast
                  with confidence intervals and regime breakdown.
                </p>
              </div>
              <div className="rounded-lg border border-gray-100 p-4">
                <h4 className="font-semibold text-gray-800 mb-1">3. Understand why</h4>
                <p className="text-gray-500">
                  View correlated events and historical analogs that
                  contextualize the forecast trajectory.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
