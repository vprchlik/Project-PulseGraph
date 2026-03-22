"use client";

import type { RegimeProbs } from "@/lib/api";
import clsx from "clsx";

interface Props {
  regimeProbs: RegimeProbs;
  dominant: string;
}

const REGIME_CONFIG: Record<string, { label: string; color: string; bg: string }> = {
  breakout: { label: "Breakout", color: "bg-emerald-500", bg: "bg-emerald-50 text-emerald-700" },
  steady_growth: { label: "Steady Growth", color: "bg-blue-500", bg: "bg-blue-50 text-blue-700" },
  plateau: { label: "Plateau", color: "bg-amber-500", bg: "bg-amber-50 text-amber-700" },
  spike_and_fade: { label: "Spike & Fade", color: "bg-orange-500", bg: "bg-orange-50 text-orange-700" },
  decay: { label: "Decay", color: "bg-red-500", bg: "bg-red-50 text-red-700" },
};

export function RegimeBar({ regimeProbs, dominant }: Props) {
  const entries = Object.entries(regimeProbs)
    .filter(([, v]) => v > 0.01)
    .sort(([, a], [, b]) => b - a);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
        Regime Probabilities
      </h3>

      <div className="flex h-4 rounded-full overflow-hidden mb-4">
        {entries.map(([regime, prob]) => {
          const cfg = REGIME_CONFIG[regime];
          return (
            <div
              key={regime}
              className={clsx(cfg?.color ?? "bg-gray-400")}
              style={{ width: `${prob * 100}%` }}
              title={`${cfg?.label ?? regime}: ${(prob * 100).toFixed(0)}%`}
            />
          );
        })}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {entries.map(([regime, prob]) => {
          const cfg = REGIME_CONFIG[regime];
          const isDominant = regime === dominant;
          return (
            <div
              key={regime}
              className={clsx(
                "rounded-lg px-3 py-2 text-sm border",
                isDominant
                  ? `${cfg?.bg} border-current font-semibold`
                  : "bg-gray-50 text-gray-600 border-transparent"
              )}
            >
              <div className="font-medium">{cfg?.label ?? regime}</div>
              <div className="text-lg tabular-nums">{(prob * 100).toFixed(0)}%</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
