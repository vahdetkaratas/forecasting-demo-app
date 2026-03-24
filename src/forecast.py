from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _normalize_freq(freq: str) -> str:
    if freq in {"M", "MS"}:
        return "M"
    return freq


def _hw_seasonal_spec(freq: str, n_obs: int) -> tuple[str | None, int | None, str]:
    """
    Choose Holt-Winters seasonal periods from resampled frequency and history length.
    Returns (seasonal_mode, seasonal_periods, label for meta).
    """
    f = _normalize_freq(freq)
    if f == "D":
        if n_obs >= 28:
            return "add", 7, "weekly_cycle"
        return None, None, "none"
    if f == "W":
        if n_obs >= 104:
            return "add", 52, "yearly_weeks"
        return None, None, "none"
    if f == "M":
        if n_obs >= 36:
            return "add", 12, "yearly_months"
        return None, None, "none"
    return None, None, "none"


def _classify_fit_failure(exc: BaseException) -> tuple[str, str]:
    """
    Map exception to (fallback_reason, user_safe_message).
    fallback_reason is stable for JSON; message is truncated for UI/metadata.
    """
    msg = f"{type(exc).__name__}: {exc}"
    msg = msg.replace("\n", " ").strip()[:280]
    if isinstance(exc, ImportError):
        return "statsmodels_import_error", msg
    if isinstance(exc, (ValueError, TypeError)):
        return "fit_invalid_input", msg
    if isinstance(exc, np.linalg.LinAlgError):
        return "fit_numerical_error", msg
    return "model_error", msg


def run_backtest(ts_df: pd.DataFrame, holdout: int = 14, freq: str = "D") -> dict:
    """Simple holdout backtest with model and naive baseline."""
    freq = _normalize_freq(freq)
    if len(ts_df) < holdout + 10:
        return {
            "mape": math.nan,
            "mae": math.nan,
            "baseline_mape": math.nan,
            "baseline_mae": math.nan,
            "n_test": 0,
            "model_status": "insufficient_data",
            "fit_fallback_reason": None,
            "fit_error_message": None,
        }

    train = ts_df.iloc[:-holdout].copy()
    test = ts_df.iloc[-holdout:].copy()

    y_train = train["value"].values.astype(float)
    y_test = test["value"].values.astype(float)

    baseline_pred = np.full(holdout, y_train[-1], dtype=float)
    baseline_mape = mape(y_test, baseline_pred)
    baseline_mae = mae(y_test, baseline_pred)

    fit_fallback_reason: str | None = None
    fit_error_message: str | None = None
    model_status = "naive_fallback"

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        sea, sp, _ = _hw_seasonal_spec(freq, len(y_train))
        kw: dict[str, Any] = {"trend": "add"}
        if sea is not None and sp is not None:
            kw["seasonal"] = sea
            kw["seasonal_periods"] = sp
            model_status = "holt_winters_seasonal"
        else:
            kw["seasonal"] = None
            model_status = "holt_winters_trend_only"

        model = ExponentialSmoothing(y_train, **kw).fit(optimized=True)
        y_pred = np.asarray(model.forecast(holdout), dtype=float)
    except Exception as exc:
        y_pred = baseline_pred
        model_status = "naive_fallback"
        fit_fallback_reason, fit_error_message = _classify_fit_failure(exc)

    return {
        "mape": mape(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "baseline_mape": baseline_mape,
        "baseline_mae": baseline_mae,
        "n_test": holdout,
        "model_status": model_status,
        "fit_fallback_reason": fit_fallback_reason,
        "fit_error_message": fit_error_message,
    }


def run_rolling_backtest(
    ts_df: pd.DataFrame,
    holdout: int = 14,
    splits: int = 3,
    freq: str = "D",
) -> dict:
    """
    Rolling backtest with fixed holdout window.

    Uses the last `splits` windows and returns average metrics.
    """
    n = len(ts_df)
    if splits <= 0:
        return {
            "mape": math.nan,
            "mae": math.nan,
            "baseline_mape": math.nan,
            "baseline_mae": math.nan,
            "splits_used": 0,
            "n_test": 0,
            "fit_fallback_reason": None,
            "fit_error_message": None,
        }

    required = holdout + 10 + (splits - 1) * holdout
    if n < required:
        return {
            "mape": math.nan,
            "mae": math.nan,
            "baseline_mape": math.nan,
            "baseline_mae": math.nan,
            "splits_used": 0,
            "n_test": 0,
            "fit_fallback_reason": None,
            "fit_error_message": None,
        }

    mapes: list[float] = []
    maes: list[float] = []
    b_mapes: list[float] = []
    b_maes: list[float] = []
    first_fb_reason: str | None = None
    first_fb_msg: str | None = None

    for i in range(splits):
        end_idx = n - (splits - 1 - i) * holdout
        window_df = ts_df.iloc[:end_idx].copy()
        one = run_backtest(window_df, holdout=holdout, freq=freq)
        if one["n_test"] == 0:
            continue
        mapes.append(float(one["mape"]))
        maes.append(float(one["mae"]))
        b_mapes.append(float(one["baseline_mape"]))
        b_maes.append(float(one["baseline_mae"]))
        if one.get("fit_fallback_reason") and first_fb_reason is None:
            first_fb_reason = one.get("fit_fallback_reason")
            first_fb_msg = one.get("fit_error_message")

    if not mapes:
        return {
            "mape": math.nan,
            "mae": math.nan,
            "baseline_mape": math.nan,
            "baseline_mae": math.nan,
            "splits_used": 0,
            "n_test": 0,
            "fit_fallback_reason": None,
            "fit_error_message": None,
        }

    return {
        "mape": float(np.nanmean(mapes)),
        "mae": float(np.nanmean(maes)),
        "baseline_mape": float(np.nanmean(b_mapes)),
        "baseline_mae": float(np.nanmean(b_maes)),
        "splits_used": len(mapes),
        "n_test": holdout,
        "fit_fallback_reason": first_fb_reason,
        "fit_error_message": first_fb_msg,
    }


def forecast_series(
    ts_df: pd.DataFrame,
    horizon: int = 30,
    confidence: float = 0.95,
    freq: str = "D",
) -> tuple[pd.DataFrame, dict]:
    """
    Forecast future values.

    Output columns: date, yhat, yhat_lower, yhat_upper
    """
    freq = _normalize_freq(freq)
    history = ts_df.copy()
    history = history.sort_values("date")
    y = history["value"].values.astype(float)
    last_date = history["date"].max()

    if len(history) < 14:
        forecast_values = np.full(horizon, y[-1], dtype=float)
        residual_std = float(np.std(np.diff(y))) if len(y) > 1 else float(np.std(y))
        model_status = "naive_short_history"
        fallback_reason = "insufficient_history"
        hw_seasonality = "n/a"
        fit_error_message = None
    else:
        fit_error_message = None
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            sea, sp, hw_seasonality = _hw_seasonal_spec(freq, len(y))
            kw: dict[str, Any] = {"trend": "add"}
            if sea is not None and sp is not None:
                kw["seasonal"] = sea
                kw["seasonal_periods"] = sp
                model_status = "holt_winters_seasonal"
            else:
                kw["seasonal"] = None
                model_status = "holt_winters_trend_only"

            model = ExponentialSmoothing(y, **kw).fit(optimized=True)
            forecast_values = np.asarray(model.forecast(horizon), dtype=float)
            fitted = np.asarray(model.fittedvalues, dtype=float)
            residual_std = float(np.std(y - fitted))
            fallback_reason = None
        except Exception as exc:
            forecast_values = np.full(horizon, y[-1], dtype=float)
            residual_std = float(np.std(np.diff(y))) if len(y) > 1 else float(np.std(y))
            model_status = "naive_fallback"
            fallback_reason, fit_error_message = _classify_fit_failure(exc)
            hw_seasonality = "n/a"

    z = 1.96 if confidence >= 0.95 else 1.64
    band = max(1e-6, z * residual_std)

    effective_freq = "MS" if freq == "M" else freq
    freq_offset = to_offset(effective_freq)
    future_dates = pd.date_range(last_date + freq_offset, periods=horizon, freq=effective_freq)
    output = pd.DataFrame(
        {
            "date": future_dates,
            "yhat": forecast_values,
            "yhat_lower": forecast_values - band,
            "yhat_upper": forecast_values + band,
        }
    )
    meta = {
        "model_status": model_status,
        "fallback_reason": fallback_reason,
        "hw_seasonality": hw_seasonality if len(history) >= 14 else "n/a",
        "fit_error_message": fit_error_message,
    }
    return output, meta
