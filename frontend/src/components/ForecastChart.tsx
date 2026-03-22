"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastData } from "@/lib/api";

interface Props {
  forecast: ForecastData;
}

export function ForecastChart({ forecast }: Props) {
  const data = forecast.median_forecast.map((med, i) => ({
    day: i + 1,
    median: Math.round(med * 100) / 100,
    lower95: Math.round(forecast.lower_95[i] * 100) / 100,
    upper95: Math.round(forecast.upper_95[i] * 100) / 100,
    lower80: Math.round(forecast.lower_80[i] * 100) / 100,
    upper80: Math.round(forecast.upper_80[i] * 100) / 100,
  }));

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
        {forecast.horizon}-Day Forecast
      </h3>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="day"
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            label={{ value: "Days ahead", position: "insideBottom", offset: -2, fontSize: 12 }}
          />
          <YAxis tick={{ fontSize: 12, fill: "#94a3b8" }} />
          <Tooltip
            contentStyle={{
              borderRadius: "8px",
              border: "1px solid #e2e8f0",
              fontSize: "13px",
            }}
          />
          <Area
            type="monotone"
            dataKey="upper95"
            stackId="band95"
            stroke="none"
            fill="#dbeafe"
            fillOpacity={0.3}
            name="95% upper"
          />
          <Area
            type="monotone"
            dataKey="lower95"
            stackId="band95"
            stroke="none"
            fill="#ffffff"
            fillOpacity={1}
            name="95% lower"
          />
          <Area
            type="monotone"
            dataKey="upper80"
            stackId="band80"
            stroke="none"
            fill="#bfdbfe"
            fillOpacity={0.4}
            name="80% upper"
          />
          <Area
            type="monotone"
            dataKey="lower80"
            stackId="band80"
            stroke="none"
            fill="#ffffff"
            fillOpacity={1}
            name="80% lower"
          />
          <Area
            type="monotone"
            dataKey="median"
            stroke="#4263eb"
            strokeWidth={2}
            fill="none"
            name="Median forecast"
          />
        </AreaChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 mt-2">
        Model: {forecast.model_name} &middot; Context: {forecast.context_length} days
      </p>
    </div>
  );
}
