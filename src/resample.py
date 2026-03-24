from __future__ import annotations

import pandas as pd


def infer_frequency(date_series: pd.Series) -> str:
    """Infer a practical frequency label for resampling."""
    parsed = pd.to_datetime(date_series, errors="coerce").dropna().sort_values()
    if parsed.shape[0] < 3:
        return "D"

    deltas = parsed.diff().dropna()
    median_days = float(deltas.dt.total_seconds().median() / 86400.0)

    if median_days <= 1.5:
        return "D"
    if median_days <= 10:
        return "W"
    return "M"


def prepare_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "D",
) -> tuple[pd.DataFrame, dict]:
    """
    Clean and resample an input dataframe into a regular time series.

    Returns:
        (prepared_df, summary_dict)
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found.")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found.")
    if freq not in {"auto", "D", "W", "M"}:
        raise ValueError("Invalid frequency. Expected: auto, D, W, M.")

    local = df[[date_col, value_col]].copy()
    original_rows = len(local)
    if original_rows == 0:
        raise ValueError("CSV is empty. Aim for at least 14–30 time-series rows.")

    local[date_col] = pd.to_datetime(local[date_col], errors="coerce")
    invalid_dates = int(local[date_col].isna().sum())
    local = local.dropna(subset=[date_col])

    local[value_col] = pd.to_numeric(local[value_col], errors="coerce")
    invalid_values = int(local[value_col].isna().sum())
    local = local.dropna(subset=[value_col])
    if local.empty:
        raise ValueError("No rows left with both a valid date and numeric value.")

    duplicate_rows = int(local.duplicated(subset=[date_col]).sum())
    local = local.sort_values(date_col)

    grouped = local.groupby(date_col, as_index=False)[value_col].sum()
    inferred_freq = infer_frequency(grouped[date_col])
    use_freq = inferred_freq if freq == "auto" else freq
    internal_freq = "MS" if use_freq == "M" else use_freq
    series = grouped.set_index(date_col)[value_col].resample(internal_freq).sum()
    missing_periods = int(series.isna().sum())
    had_gaps = missing_periods > 0

    # Safer than silent zeros: time-linear interpolation between observed bucket totals,
    # then carry last known forward / first known backward; only then zero if still NaN.
    series = series.interpolate(method="time", limit_direction="both")
    series = series.ffill().bfill()
    still_na = int(series.isna().sum())
    zero_fill_remaining = 0
    if still_na > 0:
        series = series.fillna(0.0)
        zero_fill_remaining = still_na

    prepared = series.reset_index().rename(columns={date_col: "date", value_col: "value"})
    if prepared.empty:
        raise ValueError("Prepared series is empty. Check your CSV format.")
    if prepared["date"].is_monotonic_increasing is False:
        prepared = prepared.sort_values("date").reset_index(drop=True)
    if prepared["date"].nunique() < 10:
        raise ValueError("Series too short: need at least 10 unique dates.")

    summary = {
        "original_rows": int(original_rows),
        "invalid_dates": invalid_dates,
        "invalid_values": invalid_values,
        "duplicate_rows": duplicate_rows,
        "missing_periods_after_resample": missing_periods,
        "had_resample_gaps": bool(had_gaps),
        "resample_gap_strategy": (
            "time_interpolate_ffill_bfill"
            + ("_then_zero_remaining" if zero_fill_remaining else "")
        ),
        "zero_filled_periods_after_imputation": int(zero_fill_remaining),
        "final_points": int(len(prepared)),
        "detected_frequency": inferred_freq,
        "used_frequency": use_freq,
        "date_min": str(prepared["date"].min().date()) if not prepared.empty else None,
        "date_max": str(prepared["date"].max().date()) if not prepared.empty else None,
        "is_short_series": bool(len(prepared) < 30),
    }

    return prepared, summary
