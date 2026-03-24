from __future__ import annotations

import pandas as pd

DATE_CANDIDATE_NAMES = {
    "date",
    "datetime",
    "timestamp",
    "time",
    "ds",
    "day",
}

VALUE_CANDIDATE_NAMES = {
    "value",
    "sales",
    "revenue",
    "demand",
    "visitors",
    "orders",
    "y",
}


def _is_datetime_like(series: pd.Series) -> bool:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean() >= 0.7


def detect_time_column(df: pd.DataFrame) -> str:
    """Detect the most likely time/date column."""
    if df.empty:
        raise ValueError("CSV appears empty. Upload at least a few rows.")

    lowered = {c.lower(): c for c in df.columns}
    for key in DATE_CANDIDATE_NAMES:
        if key in lowered:
            return lowered[key]

    for col in df.columns:
        if _is_datetime_like(df[col]):
            return col

    raise ValueError(
        "No date column found. Add a column such as 'date', 'datetime', or 'timestamp'."
    )


def detect_value_column(df: pd.DataFrame, date_col: str) -> str:
    """Detect target/value column for forecasting."""
    candidates = [c for c in df.columns if c != date_col]
    if not candidates:
        raise ValueError(
            "No column besides the date column was found. Forecasting needs a numeric value column."
        )
    lowered = {c.lower(): c for c in candidates}
    for key in VALUE_CANDIDATE_NAMES:
        if key in lowered:
            return lowered[key]

    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_candidates:
        return numeric_candidates[0]

    raise ValueError(
        "No numeric value column found. Expected something like 'value', 'sales', 'demand', or 'visitors'."
    )

