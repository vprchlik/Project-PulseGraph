"use client";

import { useState } from "react";
import { searchRegime, type RegimeSearchResult } from "@/lib/api";
import { Search } from "lucide-react";
import clsx from "clsx";

const REGIMES = [
  { value: "breakout", label: "Breakout" },
  { value: "steady_growth", label: "Steady Growth" },
  { value: "plateau", label: "Plateau" },
  { value: "spike_and_fade", label: "Spike & Fade" },
  { value: "decay", label: "Decay" },
];

const HORIZONS = [7, 30, 90];

interface Props {
  onSelectRepo?: (repoName: string) => void;
}

export function SearchPanel({ onSelectRepo }: Props) {
  const [regime, setRegime] = useState("breakout");
  const [horizon, setHorizon] = useState(30);
  const [minProb, setMinProb] = useState(0.3);
  const [results, setResults] = useState<RegimeSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    try {
      const data = await searchRegime(regime, horizon, minProb);
      setResults(data.results);
      setSearched(true);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
        Future Regime Search
      </h3>

      <div className="flex flex-wrap gap-3 mb-4">
        <select
          value={regime}
          onChange={(e) => setRegime(e.target.value)}
          className="rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white"
        >
          {REGIMES.map((r) => (
            <option key={r.value} value={r.value}>{r.label}</option>
          ))}
        </select>

        <select
          value={horizon}
          onChange={(e) => setHorizon(Number(e.target.value))}
          className="rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white"
        >
          {HORIZONS.map((h) => (
            <option key={h} value={h}>{h} days</option>
          ))}
        </select>

        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-500">Min probability:</label>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={minProb}
            onChange={(e) => setMinProb(Number(e.target.value))}
            className="w-24"
          />
          <span className="text-sm tabular-nums text-gray-600 w-10">
            {(minProb * 100).toFixed(0)}%
          </span>
        </div>

        <button
          onClick={handleSearch}
          disabled={loading}
          className={clsx(
            "flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-white",
            loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-pulse-600 hover:bg-pulse-700 transition-colors"
          )}
        >
          <Search size={16} />
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {searched && (
        <div>
          <p className="text-sm text-gray-500 mb-3">
            {results.length} {results.length === 1 ? "repo" : "repos"} with{" "}
            {">"}{(minProb * 100).toFixed(0)}% {regime.replace("_", " ")} probability
          </p>

          {results.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 text-left text-gray-500">
                    <th className="pb-2 font-medium">Repository</th>
                    <th className="pb-2 font-medium">Probability</th>
                    <th className="pb-2 font-medium">Language</th>
                    <th className="pb-2 font-medium hidden sm:table-cell">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r) => (
                    <tr
                      key={r.repo_name}
                      className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors"
                      onClick={() => onSelectRepo?.(r.repo_name)}
                    >
                      <td className="py-2.5 font-medium text-pulse-700">{r.repo_name}</td>
                      <td className="py-2.5 tabular-nums">{(r.regime_prob * 100).toFixed(0)}%</td>
                      <td className="py-2.5 text-gray-500">{r.language || "—"}</td>
                      <td className="py-2.5 text-gray-400 hidden sm:table-cell truncate max-w-xs">
                        {r.description || "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
