from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import src.forecast as forecast_mod
from src.forecast import (
    _hw_seasonal_spec,
    forecast_series,
    mae,
    mape,
    run_backtest,
    run_rolling_backtest,
)
from src.explain import build_explanation
from src.resample import infer_frequency, prepare_series
from src.time_detect import detect_time_column, detect_value_column


def test_mae_non_negative() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 29.0])
    assert mae(y_true, y_pred) >= 0.0


def test_mape_non_negative() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 28.0])
    assert mape(y_true, y_pred) >= 0.0


def test_infer_frequency_daily() -> None:
    dates = pd.Series(pd.date_range("2025-01-01", periods=10, freq="D"))
    assert infer_frequency(dates) == "D"


def test_infer_frequency_weekly() -> None:
    dates = pd.Series(pd.date_range("2025-01-01", periods=10, freq="W"))
    assert infer_frequency(dates) == "W"


def test_infer_frequency_monthly() -> None:
    dates = pd.Series(pd.date_range("2025-01-01", periods=10, freq="MS"))
    assert infer_frequency(dates) == "M"


def test_backtest_outputs_baseline_keys() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=35, freq="D"),
            "value": np.linspace(10, 20, 35),
        }
    )
    metrics = run_backtest(frame, holdout=14)
    assert "baseline_mape" in metrics
    assert "baseline_mae" in metrics
    assert "model_status" in metrics


def test_rolling_backtest_returns_split_info() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "value": np.linspace(10, 30, 80),
        }
    )
    metrics = run_rolling_backtest(frame, holdout=10, splits=3)
    assert metrics["splits_used"] == 3
    assert metrics["n_test"] == 10


def test_explanation_returns_text() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=30, freq="D"),
            "value": np.linspace(100, 130, 30),
        }
    )
    future = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-31", periods=7, freq="D"),
            "yhat": np.linspace(131, 138, 7),
            "yhat_lower": np.linspace(128, 135, 7),
            "yhat_upper": np.linspace(134, 141, 7),
        }
    )
    text = build_explanation(history, future, freq="D")
    assert isinstance(text, str)
    assert len(text) > 20


def test_forecast_series_weekly_frequency_steps() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20, freq="W"),
            "value": np.linspace(100, 160, 20),
        }
    )
    out, meta = forecast_series(history, horizon=4, confidence=0.95, freq="W")
    diffs = out["date"].diff().dropna().dt.days.tolist()
    assert diffs == [7, 7, 7]
    assert "model_status" in meta


def test_forecast_series_monthly_frequency_steps() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "value": np.linspace(50, 95, 12),
        }
    )
    out, _ = forecast_series(history, horizon=3, confidence=0.95, freq="M")
    assert out["date"].dt.month.tolist() == [1, 2, 3]


def test_detect_time_column_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        detect_time_column(pd.DataFrame())


def test_detect_value_column_raises_on_no_numeric() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "label": ["a", "b"],
        }
    )
    with pytest.raises(ValueError):
        detect_value_column(frame, "date")


def test_prepare_series_raises_on_invalid_freq() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=12, freq="D"),
            "value": np.arange(12),
        }
    )
    with pytest.raises(ValueError):
        prepare_series(frame, "date", "value", freq="H")


def test_hw_seasonal_spec_daily_weekly_cycle_threshold() -> None:
    sea, sp, label = _hw_seasonal_spec("D", 28)
    assert sea == "add" and sp == 7 and label == "weekly_cycle"


def test_hw_seasonal_spec_daily_trend_only_below_threshold() -> None:
    sea, sp, label = _hw_seasonal_spec("D", 27)
    assert sea is None and sp is None and label == "none"


def test_backtest_short_training_avoids_seasonal_status() -> None:
    """Train length 35-14=21 < 28 → Holt-Winters without seasonal terms (or naive if fit fails)."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=35, freq="D"),
            "value": np.linspace(10, 28, 35),
        }
    )
    metrics = run_backtest(frame, holdout=14, freq="D")
    assert metrics["model_status"] in ("holt_winters_trend_only", "naive_fallback")
    assert metrics["model_status"] != "holt_winters_seasonal"


def test_backtest_long_training_allows_seasonal_when_fit_succeeds() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=90, freq="D"),
            "value": np.linspace(100, 220, 90) + 5 * np.sin(np.linspace(0, 4 * np.pi, 90)),
        }
    )
    metrics = run_backtest(frame, holdout=14, freq="D")
    assert metrics["model_status"] in (
        "holt_winters_seasonal",
        "holt_winters_trend_only",
        "naive_fallback",
    )
    if metrics["model_status"] != "naive_fallback":
        assert metrics["model_status"] == "holt_winters_seasonal"


def test_rolling_backtest_nanmean_ignores_nan_split(monkeypatch: pytest.MonkeyPatch) -> None:
    def split(
        mape_v: float,
        mae_v: float,
        b_mape: float,
        b_mae: float,
    ) -> dict:
        return {
            "mape": mape_v,
            "mae": mae_v,
            "baseline_mape": b_mape,
            "baseline_mae": b_mae,
            "n_test": 14,
            "model_status": "x",
            "fit_fallback_reason": None,
            "fit_error_message": None,
        }

    side_effect = [
        split(10.0, 1.0, 12.0, 2.0),
        split(float("nan"), float("nan"), float("nan"), float("nan")),
        split(20.0, 2.0, 22.0, 3.0),
    ]
    monkeypatch.setattr(forecast_mod, "run_backtest", MagicMock(side_effect=side_effect))
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=60, freq="D"),
            "value": np.arange(60, dtype=float),
        }
    )
    out = run_rolling_backtest(frame, holdout=14, splits=3, freq="D")
    assert out["splits_used"] == 3
    assert out["mape"] == pytest.approx(15.0)
    assert out["mae"] == pytest.approx(1.5)

