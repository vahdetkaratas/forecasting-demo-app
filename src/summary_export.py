"""Build JSON-serializable summary dict for downloads (no Streamlit)."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .demo_scenario import DEMO_SCENARIO_CAPTION
except ImportError:
    from demo_scenario import DEMO_SCENARIO_CAPTION


def build_summary_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle["summary"]
    metrics = bundle["metrics"]
    forecast_meta = bundle["forecast_meta"]
    rolling_mode = bundle["rolling_mode"]
    rolling_splits = bundle["rolling_splits"]
    payload: dict[str, Any] = {
        "horizon": int(bundle["horizon"]),
        "holdout_effective": int(bundle["effective_holdout"]),
        "rolling_mode": bool(rolling_mode),
        "rolling_splits": int(rolling_splits) if rolling_mode else None,
        "frequency": summary["used_frequency"],
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "rows_input": int(summary["original_rows"]),
        "rows_prepared": int(summary["final_points"]),
        "model_status": forecast_meta["model_status"],
        "fallback_reason": forecast_meta["fallback_reason"],
        "hw_seasonality": forecast_meta.get("hw_seasonality"),
        "fit_error_message": forecast_meta.get("fit_error_message"),
        "resample": {
            "had_gaps": summary.get("had_resample_gaps", False),
            "missing_periods_after_resample": summary.get("missing_periods_after_resample", 0),
            "gap_strategy": summary.get("resample_gap_strategy"),
            "zero_filled_periods_after_imputation": summary.get(
                "zero_filled_periods_after_imputation", 0
            ),
        },
        "backtest_fit_fallback_reason": metrics.get("fit_fallback_reason"),
        "backtest_fit_error_message": metrics.get("fit_error_message"),
        "metrics": {
            "mape": None if not np.isfinite(metrics["mape"]) else float(metrics["mape"]),
            "mae": None if not np.isfinite(metrics["mae"]) else float(metrics["mae"]),
            "baseline_mape": None
            if not np.isfinite(metrics["baseline_mape"])
            else float(metrics["baseline_mape"]),
            "baseline_mae": None
            if not np.isfinite(metrics["baseline_mae"])
            else float(metrics["baseline_mae"]),
            "n_test": int(metrics["n_test"]),
        },
    }
    if rolling_mode:
        payload["metrics"]["splits_used"] = int(metrics.get("splits_used", 0))
    if bundle.get("is_demo"):
        payload["demo_scenario"] = bundle.get("demo_scenario_note") or DEMO_SCENARIO_CAPTION
    return payload
