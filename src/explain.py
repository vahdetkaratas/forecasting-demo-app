from __future__ import annotations

import numpy as np
import pandas as pd


def _seasonality_comment(history: pd.DataFrame, freq: str) -> str:
    if freq != "D" or len(history) < 21:
        return "Seasonality commentary is limited at this data frequency."

    local = history.copy()
    local["weekday"] = local["date"].dt.dayofweek
    weekday_means = local.groupby("weekday")["value"].mean()
    if weekday_means.empty:
        return "Could not read a clear weekly pattern."

    spread_ratio = float((weekday_means.max() - weekday_means.min()) / (weekday_means.mean() + 1e-9))
    if spread_ratio > 0.15:
        return "Weekly pattern is pronounced (large day-to-day level spread)."
    return "Weekly pattern is weak to moderate."


def build_explanation(history: pd.DataFrame, forecast: pd.DataFrame, freq: str = "D") -> str:
    if history.empty or forecast.empty:
        return "Not enough data to generate an explanation."

    hist_start = float(history["value"].iloc[0])
    hist_end = float(history["value"].iloc[-1])
    growth = ((hist_end - hist_start) / abs(hist_start) * 100.0) if hist_start != 0 else 0.0

    avg_future = float(forecast["yhat"].mean())
    latest_hist = float(history["value"].iloc[-1])
    delta_future = (
        ((avg_future - latest_hist) / abs(latest_hist) * 100.0) if latest_hist != 0 else 0.0
    )
    recent_window = min(14, len(history))
    recent = history["value"].tail(recent_window).values.astype(float)
    volatility = float(np.std(recent) / (np.mean(recent) + 1e-9) * 100.0)

    trend_text = "upward" if growth >= 0 else "downward"
    future_text = "increase" if delta_future >= 0 else "decrease"
    seasonality_text = _seasonality_comment(history, freq)

    return (
        f"Over the historical window the series shows an overall {trend_text} trend of about "
        f"{abs(growth):.1f}%. Relative to the latest observed value, the forecast horizon averages "
        f"about {abs(delta_future):.1f}% {future_text}. Volatility over the last {recent_window} "
        f"points is roughly {volatility:.1f}%.\n\n"
        f"Seasonality note: {seasonality_text}\n\n"
        "Limitations: This MVP uses a univariate approach. Campaigns, holidays, and external "
        "drivers are not modeled explicitly."
    )
