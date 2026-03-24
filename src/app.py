from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st

from demo_scenario import DEMO_SCENARIO_CAPTION
from explain import build_explanation
from footer_config import app_footer_text
from forecast import forecast_series, run_backtest, run_rolling_backtest
from format_metrics import format_finite_metric
from load import load_csv
from resample import prepare_series
from run_fingerprints import (
    data_fingerprint_demo,
    data_fingerprint_upload_id,
    data_fingerprint_upload_none,
    data_fingerprint_upload_sha256,
    settings_fingerprint,
)
from summary_export import build_summary_payload
from time_detect import detect_time_column, detect_value_column
from visualize import build_forecast_figure


def _render_app_footer() -> None:
    text = app_footer_text()
    if not text:
        return
    st.divider()
    st.caption(text)


def _freq_unit(freq: str) -> str:
    if freq == "W":
        return "week"
    if freq in {"M", "MS"}:
        return "month"
    return "day"


def _recommended_holdout(freq: str) -> int:
    if freq == "W":
        return 8
    if freq in {"M", "MS"}:
        return 6
    return 14


def _data_fingerprint(source_mode: str, demo_name: str | None, uploaded_file: Any) -> tuple[Any, ...]:
    if source_mode == "Demo":
        return data_fingerprint_demo(demo_name or "")
    if uploaded_file is None:
        return data_fingerprint_upload_none()
    fid = getattr(uploaded_file, "file_id", None)
    if fid is not None:
        sz = getattr(uploaded_file, "size", None)
        return data_fingerprint_upload_id(str(fid), sz if sz is not None else -1)
    return data_fingerprint_upload_sha256(hashlib.sha256(uploaded_file.getvalue()).hexdigest())


def _render_model_status(forecast_meta: dict[str, Any]) -> None:
    status = forecast_meta["model_status"]
    reason = forecast_meta.get("fallback_reason")
    hw = forecast_meta.get("hw_seasonality", "n/a")

    if status == "holt_winters_seasonal":
        labels = {
            "weekly_cycle": "7-day seasonal component",
            "yearly_weeks": "52-week seasonal component",
            "yearly_months": "12-month seasonal component",
        }
        detail = labels.get(hw, hw)
        st.success(f"Holt-Winters (with {detail}).")
    elif status == "holt_winters_trend_only":
        st.success(
            "Holt-Winters (trend only). Seasonal terms are disabled for this frequency "
            "or series length so the fit stays credible."
        )
    elif status == "naive_fallback":
        labels = {
            "statsmodels_import_error": "statsmodels could not be imported",
            "fit_invalid_input": "fit rejected the input (value/type)",
            "fit_numerical_error": "numerical failure during fit (e.g. linear algebra)",
            "model_error": "unexpected error during fit",
        }
        hint = labels.get(str(reason), "fit did not complete")
        msg = (
            f"Forecast uses a **naive last-value** fallback — Holt-Winters did not run successfully "
            f"({hint})."
        )
        st.warning(msg)
        detail = forecast_meta.get("fit_error_message")
        if detail:
            st.caption("Technical detail (truncated):")
            st.code(detail, language=None)
    elif status == "naive_short_history":
        st.info("Forecast uses the last observed value — history is shorter than the minimum for Holt-Winters here.")
    else:
        st.info(f"Model status: `{status}`")


st.set_page_config(
    page_title="Forecasting demo — sales & demand",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "forecast_bundle" not in st.session_state:
    st.session_state.forecast_bundle = None
if "run_data_fp" not in st.session_state:
    st.session_state.run_data_fp = None
if "run_settings_fp" not in st.session_state:
    st.session_state.run_settings_fp = None

st.title("Univariate demand / sales forecast")
st.markdown(
    "Upload a CSV or use a built-in demo, then generate a **forecast with uncertainty bands**, "
    "**backtest metrics** vs a naive baseline, and a short **text summary**. "
    "Built for portfolio demos — not a production forecasting platform."
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        """
1. **Data**: choose **Demo** (two sample series) or **Upload CSV** with a date column and a numeric target (`sales`, `value`, `visitors`, …).
2. **Sidebar**: set horizon, backtest options, and resample frequency (`auto` / daily / weekly / monthly).
3. Click **Run forecast** — fitting and charts run **only** then (not on every slider move).
4. Read **Model status** to see whether Holt-Winters ran with seasonality, trend-only, or a naive fallback.
5. **Download** forecast CSV / summary JSON, or save to `artifacts/forecasts/` if the server has a writable disk.
        """
    )

with st.sidebar:
    st.subheader("Forecast settings")
    horizon = st.slider("Horizon (period)", min_value=4, max_value=90, value=30, step=1)
    holdout = st.slider("Backtest holdout (period)", min_value=4, max_value=30, value=14, step=1)
    adaptive_holdout = st.toggle("Use adaptive holdout", value=True)
    rolling_mode = st.toggle("Rolling backtest", value=True)
    rolling_splits = st.slider("Rolling split count", min_value=2, max_value=6, value=3, step=1)
    freq_mode = st.selectbox(
        "Resample frequency",
        options=["auto", "D", "W", "M"],
        index=0,
    )
    run_clicked = st.button("Run forecast", type="primary")

base_dir = Path(__file__).resolve().parents[1]
demo_sales = base_dir / "data" / "demo_sales" / "sales_ts.csv"
demo_traffic = base_dir / "data" / "demo_traffic" / "traffic_ts.csv"
artifact_path = base_dir / "artifacts" / "forecasts" / "forecast.csv"
summary_artifact_path = base_dir / "artifacts" / "forecasts" / "summary.json"

demo_name: str | None = None
with st.container(border=True):
    st.subheader("Data & input")
    source_mode = st.radio("Data source", options=["Demo", "Upload CSV"], horizontal=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], disabled=source_mode != "Upload CSV")
    if source_mode == "Demo":
        demo_name = st.selectbox("Choose demo", options=["Sales demo", "Traffic demo"])

    if source_mode == "Upload CSV" and uploaded_file is None:
        st.info("Upload a CSV to start (date + value or sales/visitors style columns).")
        st.caption("No CSV handy? Download a sample below, then upload it here.")
        if demo_sales.is_file() and demo_traffic.is_file():
            d1, d2 = st.columns(2)
            d1.download_button(
                label="Download sample — sales",
                data=demo_sales.read_bytes(),
                file_name="sample_sales.csv",
                mime="text/csv",
            )
            d2.download_button(
                label="Download sample — traffic",
                data=demo_traffic.read_bytes(),
                file_name="sample_traffic.csv",
                mime="text/csv",
            )
        else:
            st.warning("Bundled sample CSVs are missing from the `data/` folder.")
        _render_app_footer()
        st.stop()

cur_data_fp = _data_fingerprint(source_mode, demo_name, uploaded_file)
cur_settings_fp = settings_fingerprint(
    horizon, holdout, adaptive_holdout, rolling_mode, rolling_splits, freq_mode
)

stale = (
    st.session_state.forecast_bundle is not None
    and (
        st.session_state.run_data_fp != cur_data_fp
        or st.session_state.run_settings_fp != cur_settings_fp
    )
)
if stale:
    st.info("Data or settings changed since the last run. Click **Run forecast** in the sidebar to refresh results.")

if run_clicked:
    try:
        if source_mode == "Demo":
            raw_df = load_csv(demo_sales if demo_name == "Sales demo" else demo_traffic)
        else:
            raw_df = load_csv(uploaded_file)
        date_col = detect_time_column(raw_df)
        value_col = detect_value_column(raw_df, date_col)
        ts_df, summary = prepare_series(raw_df, date_col, value_col, freq=freq_mode)
        used_f = summary["used_frequency"]
        effective_holdout = _recommended_holdout(used_f) if adaptive_holdout else holdout
        future, forecast_meta = forecast_series(
            ts_df, horizon=horizon, confidence=0.95, freq=used_f
        )
        if rolling_mode:
            metrics = run_rolling_backtest(
                ts_df, holdout=effective_holdout, splits=rolling_splits, freq=used_f
            )
        else:
            metrics = run_backtest(ts_df, holdout=effective_holdout, freq=used_f)

        st.session_state.forecast_bundle = {
            "ts_df": ts_df,
            "summary": summary,
            "future": future,
            "forecast_meta": forecast_meta,
            "metrics": metrics,
            "effective_holdout": effective_holdout,
            "horizon": horizon,
            "rolling_mode": rolling_mode,
            "rolling_splits": rolling_splits,
            "is_demo": source_mode == "Demo",
            "demo_scenario_note": DEMO_SCENARIO_CAPTION if source_mode == "Demo" else None,
        }
        st.session_state.run_data_fp = cur_data_fp
        st.session_state.run_settings_fp = cur_settings_fp
    except Exception as exc:
        st.error(f"Run failed: {exc}")
        st.info(
            "Expected: one datetime column and one numeric target "
            "(`value` / `sales` / `demand` / `visitors`)."
        )
        st.session_state.forecast_bundle = None
        st.session_state.run_data_fp = None
        st.session_state.run_settings_fp = None

bundle_ok = (
    st.session_state.forecast_bundle is not None
    and st.session_state.run_data_fp == cur_data_fp
    and st.session_state.run_settings_fp == cur_settings_fp
)

if not bundle_ok:
    st.info("Configure the **sidebar**, then click **Run forecast** to compute validation, chart, metrics, and downloads.")
    _render_app_footer()
    st.stop()

b = st.session_state.forecast_bundle
ts_df = b["ts_df"]
summary = b["summary"]
future = b["future"]
forecast_meta = b["forecast_meta"]
metrics = b["metrics"]
effective_holdout = b["effective_holdout"]
unit = _freq_unit(summary["used_frequency"])

with st.container(border=True):
    st.subheader("Input validation")
    summary_df = pd.DataFrame([summary]).T.reset_index()
    summary_df.columns = ["check", "value"]
    with st.expander("View validation details", expanded=False):
        st.dataframe(summary_df, use_container_width=True)

if summary.get("had_resample_gaps"):
    n_miss = int(summary.get("missing_periods_after_resample", 0))
    st.warning(
        f"Resampling left **{n_miss}** empty period(s) with no raw observations in that bucket. "
        "Values were imputed using time interpolation and forward/backward fill — "
        "not assumed to be true zeros. Check validation details for the strategy used."
    )
if int(summary.get("zero_filled_periods_after_imputation", 0)) > 0:
    nz = int(summary["zero_filled_periods_after_imputation"])
    st.error(
        f"**{nz}** period(s) still had no value after interpolation; they were set to **0** as a "
        "last resort. Treat the series as low-trust for those gaps."
    )

if len(ts_df) < 14:
    st.warning("At least 14-30 points are recommended for stable forecasts.")
elif summary.get("is_short_series", False):
    st.info("Series is short (<30 points). Uncertainty may be higher.")

st.caption(f"Horizon: {b['horizon']} {unit} | Frequency: {summary['used_frequency']}")

original_rows = max(summary["original_rows"], 1)
invalid_ratio = (summary["invalid_dates"] + summary["invalid_values"]) / original_rows
duplicate_ratio = summary["duplicate_rows"] / original_rows
if invalid_ratio > 0.2 or duplicate_ratio > 0.2:
    st.warning(
        "Data quality risk: high invalid/duplicate ratio. Forecast reliability may decrease."
    )
elif invalid_ratio > 0.1 or duplicate_ratio > 0.1:
    st.info("Data quality note: medium invalid/duplicate ratio.")

with st.container(border=True):
    st.subheader("Forecast & evaluation")
    if b.get("is_demo"):
        st.caption(DEMO_SCENARIO_CAPTION)
    left, right = st.columns([2, 1])
    with left:
        fig = build_forecast_figure(ts_df, future)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("How to read the chart"):
            st.markdown(
                """
- **Muted line**: observed history after cleaning and resampling.
- **Accent line**: point forecast (`yhat`) for future periods.
- **Shaded band**: approximate uncertainty — width is ±1.96× residual standard deviation around the point forecast (treated as a rough 95% band). It is **not** a calibrated or guaranteed prediction interval.
- Gaps or odd jumps usually mean sparse or irregular input; check **Input validation**.
                """
            )

    with right:
        st.caption(f"Backtest holdout (effective): {effective_holdout} {unit}")
        if b["rolling_mode"]:
            st.caption(f"Backtest mode: Rolling ({b['rolling_splits']} splits)")
        if metrics["n_test"] == 0:
            st.write("Not enough data to compute backtest metrics.")
        elif not np.isfinite(metrics["mape"]) or not np.isfinite(metrics["mae"]):
            st.info(
                "Backtest metrics are not available (non-finite values — often too little data "
                "for the chosen rolling setup)."
            )
        else:
            c1, c2 = st.columns(2)
            c1.metric("MAPE (%)", format_finite_metric(float(metrics["mape"])))
            c2.metric("MAE", format_finite_metric(float(metrics["mae"])))
            if (
                np.isfinite(metrics["baseline_mape"])
                and metrics["baseline_mape"] > 0
                and np.isfinite(metrics["mape"])
            ):
                improvement = (
                    (metrics["baseline_mape"] - metrics["mape"]) / metrics["baseline_mape"]
                ) * 100.0
                if np.isfinite(improvement):
                    st.metric("Model vs baseline (MAPE)", f"{improvement:.1f}%")
            st.caption(
                "Naive baseline MAPE / MAE: "
                f"{format_finite_metric(float(metrics['baseline_mape']))} / "
                f"{format_finite_metric(float(metrics['baseline_mae']))}"
            )
            st.caption(f"Holdout size: {metrics['n_test']} {unit}")
            if b["rolling_mode"]:
                st.caption(f"Splits used: {metrics['splits_used']}")
        if metrics.get("fit_error_message"):
            st.caption(
                "Backtest window fit: at least one holdout fit used the naive baseline. "
                "Last error detail (truncated):"
            )
            st.code(str(metrics["fit_error_message"]), language=None)

with st.container(border=True):
    st.subheader("Model status")
    _render_model_status(forecast_meta)

with st.container(border=True):
    st.subheader("Narrative summary")
    st.write(build_explanation(ts_df, future, freq=summary["used_frequency"]))
    st.caption(
        "This text is rule-based on your series and forecast — useful for demos, not a substitute "
        "for domain review. The chart band reflects model residual spread, not a hard bound."
    )

with st.container(border=True):
    st.subheader("Forecast output")
    download_df = future.copy()
    st.dataframe(download_df, use_container_width=True, height=320)
    st.download_button(
        label="Download forecast CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast.csv",
        mime="text/csv",
        key="dl_forecast_csv",
    )
    summary_bytes = json.dumps(build_summary_payload(b), ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label="Download summary JSON",
        data=summary_bytes,
        file_name="summary.json",
        mime="application/json",
        key="dl_summary_json",
    )
    if st.button("Save artifacts to project folder"):
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            download_df.to_csv(artifact_path, index=False)
            summary_artifact_path.write_text(
                json.dumps(build_summary_payload(b), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success(f"Wrote `{artifact_path}` and `{summary_artifact_path}`.")
        except OSError as oe:
            st.error(f"Could not write artifacts: {oe}")

_render_app_footer()
