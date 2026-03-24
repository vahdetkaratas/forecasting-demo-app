from __future__ import annotations

import math

import pytest

from src.demo_scenario import DEMO_SCENARIO_CAPTION
from src.summary_export import build_summary_payload


def _minimal_bundle(**overrides: object) -> dict:
    b: dict = {
        "horizon": 7,
        "effective_holdout": 14,
        "rolling_mode": False,
        "rolling_splits": 3,
        "summary": {
            "used_frequency": "D",
            "date_min": "2025-01-01",
            "date_max": "2025-03-31",
            "original_rows": 90,
            "final_points": 90,
            "had_resample_gaps": False,
            "missing_periods_after_resample": 0,
            "resample_gap_strategy": None,
            "zero_filled_periods_after_imputation": 0,
        },
        "metrics": {
            "mape": 1.25,
            "mae": 2.5,
            "baseline_mape": 1.5,
            "baseline_mae": 3.0,
            "n_test": 14,
            "fit_fallback_reason": None,
            "fit_error_message": None,
        },
        "forecast_meta": {
            "model_status": "holt_winters_seasonal",
            "fallback_reason": None,
            "hw_seasonality": "weekly_cycle",
            "fit_error_message": None,
        },
        "is_demo": False,
        "demo_scenario_note": None,
    }
    b.update(overrides)
    return b


def test_build_summary_payload_maps_non_finite_metrics_to_none() -> None:
    m = _minimal_bundle()
    m["metrics"] = {
        **m["metrics"],
        "mape": float("nan"),
        "mae": float("nan"),
        "baseline_mape": float("nan"),
        "baseline_mae": float("nan"),
    }
    p = build_summary_payload(m)
    assert p["metrics"]["mape"] is None
    assert p["metrics"]["mae"] is None
    assert p["metrics"]["baseline_mape"] is None
    assert p["metrics"]["baseline_mae"] is None


def test_build_summary_payload_includes_demo_scenario_when_demo() -> None:
    p = build_summary_payload(_minimal_bundle(is_demo=True, demo_scenario_note=None))
    assert p.get("demo_scenario") == DEMO_SCENARIO_CAPTION


def test_build_summary_payload_omits_demo_scenario_for_upload_mode() -> None:
    p = build_summary_payload(_minimal_bundle(is_demo=False))
    assert "demo_scenario" not in p


def test_build_summary_payload_finite_metrics_round_trip() -> None:
    p = build_summary_payload(_minimal_bundle())
    assert p["metrics"]["mape"] == pytest.approx(1.25)
    assert math.isfinite(p["metrics"]["mae"])
