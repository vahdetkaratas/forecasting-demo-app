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
        st.caption(
            "Plain language: the app detected **repeating calendar patterns** (e.g. weekdays) and used them "
            "together with overall up/down movement to extend your series."
        )
    elif status == "holt_winters_trend_only":
        st.success(
            "Holt-Winters (trend only). Seasonal terms are disabled for this frequency "
            "or series length so the fit stays credible."
        )
        st.caption(
            "Plain language: the forecast follows **overall growth or decline** without assuming a strong weekly/monthly repeat — "
            "often because the history is short or the time step doesn’t support seasonality here."
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
        st.caption(
            "Plain language: the usual forecast model **did not complete**, so the app shows a **flat line at the last known value** "
            "and warns you — the chart is still usable, but treat it as a placeholder."
        )
        detail = forecast_meta.get("fit_error_message")
        if detail:
            st.caption("Technical detail (truncated):")
            st.code(detail, language=None)
    elif status == "naive_short_history":
        st.info("Forecast uses the last observed value — history is shorter than the minimum for Holt-Winters here.")
        st.caption(
            "Plain language: **not enough history** for the main model here — the app repeats the **latest number** forward."
        )
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
    "**You don’t need a forecasting background.** If you have **dates** and **one number per day/week/month** "
    "(sales, visits, orders…), this app draws your **past**, projects a **future line**, and scores how plausible "
    "that projection is on **recent history**. Try the **Demo** or upload a CSV, then use **Run forecast** in the sidebar."
)
st.caption(
    "Portfolio demo — not a full planning system (no promotions, holidays, or external drivers modeled)."
)

st.info(
    "**What this page does:** walks a single metric from **raw CSV → cleaned timeline → forecast chart → honest score** "
    "on recent data you “hide” from the model.\n\n"
    "**Why results don’t update on every slider move:** so the UI stays fast and each chart always matches **one deliberate "
    "set of choices** — easier to explain in a demo or stakeholder review.\n\n"
    "**Where to look after you run:** **Forecast & evaluation** for the story in the chart, **Model status** for what math ran, "
    "**Narrative summary** for plain-language commentary, **Forecast output** to export files."
)

with st.expander("Read more — what each part of the app is for (and why we built it that way)", expanded=False):
    st.markdown(
        """
##### The idea in one sentence
Your past values contain **shape** (trend, sometimes weekly/monthly repeats). We **continue that shape** forward, then **check ourselves**
by pretending we don’t know the last few points and measuring error — so you see both a picture *and* a sanity check.

##### What goes in
- **CSV:** one datetime column + one numeric column (or the built-in **Demo**).
- **Why we insist on a clean time axis:** irregular timestamps or mixed gaps would make “one day ahead” ambiguous; resampling fixes that
  so the forecast matches how people actually plan (daily / weekly / monthly).

##### What comes out
- **Chart:** history + future line + shaded band. The band is a **rough** uncertainty hint from past fit errors — **not** a guarantee.
- **Metrics (MAPE / MAE):** “how wrong were we, in % and in your units?” compared to a **baseline** that always repeats the last value —
  so you can say whether the model earned its complexity.
- **Model status:** transparency. If data are short or the fit fails, we **say so** and fall back instead of silently showing a fancy curve.

##### Sidebar choices — why they exist
- **Horizon:** how far you ask the model to look ahead (longer = usually harder; errors compound).
- **Holdout / rolling:** backtest design. **Holdout** = length of the “exam.” **Rolling** = average several exams so one lucky window doesn’t fool you.
- **Adaptive holdout:** picks a sensible exam length for daily vs weekly vs monthly so you don’t need domain defaults memorized.
- **Resample frequency:** aligns the series to a regular calendar rhythm; **auto** guesses from your gaps.

##### How to use (steps)
1. **Data:** Demo or upload CSV (`date` + `sales` / `value` / `visitors` style columns).
2. **Sidebar:** adjust horizon, backtest options, frequency — read the **?** tooltips on each control for the “why.”
3. **Run forecast** — compute once, review all sections below.
4. **Download** CSV / JSON (or save to disk if the server allows).

**Jargon:** *MAPE* ≈ average percent error · *MAE* ≈ average error in your units · *baseline* = “always guess the last value.”
        """
    )

with st.sidebar:
    st.subheader("Forecast settings")
    st.caption(
        "Heavy work runs only when you click **Run forecast** — so tweaking sliders never spams the server or mixes old/new results."
    )
    horizon = st.slider(
        "Horizon (period)",
        min_value=4,
        max_value=90,
        value=30,
        step=1,
        help="How many future periods to draw. Longer horizons are harder: uncertainty grows and the model has more room to drift.",
    )
    holdout = st.slider(
        "Backtest holdout (period)",
        min_value=4,
        max_value=30,
        value=14,
        step=1,
        help="When adaptive holdout is off: how many recent periods we hide to score the model. Mimics ‘I only knew the past — how well would I have predicted the next stretch?’",
    )
    adaptive_holdout = st.toggle(
        "Use adaptive holdout",
        value=True,
        help="Uses recommended holdout lengths per frequency (e.g. 14 days, 8 weeks, 6 months) so defaults stay sensible without extra expertise.",
    )
    rolling_mode = st.toggle(
        "Rolling backtest",
        value=True,
        help="Runs several overlapping backtests and averages the scores. Why: one single window can be lucky or unlucky; averaging is more representative.",
    )
    rolling_splits = st.slider(
        "Rolling split count",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
        help="More splits → smoother average metric, but needs more history. Each split needs enough points before the holdout window.",
    )
    freq_mode = st.selectbox(
        "Resample frequency",
        options=["auto", "D", "W", "M"],
        index=0,
        help="Forces regular daily / weekly / monthly buckets (or auto-detect). Why: the forecast step size must match how you want to read the future.",
    )
    run_clicked = st.button(
        "Run forecast",
        type="primary",
        help="Runs validation, resampling, fit, backtest, and charts for the current data + all sidebar settings together.",
    )

base_dir = Path(__file__).resolve().parents[1]
demo_sales = base_dir / "data" / "demo_sales" / "sales_ts.csv"
demo_traffic = base_dir / "data" / "demo_traffic" / "traffic_ts.csv"
artifact_path = base_dir / "artifacts" / "forecasts" / "forecast.csv"
summary_artifact_path = base_dir / "artifacts" / "forecasts" / "summary.json"

demo_name: str | None = None
with st.container(border=True):
    st.subheader("Data & input")
    st.caption(
        "Pick **Demo** to explore instantly, or **Upload CSV** for your own series. **Why one numeric target:** this MVP models "
        "**one line at a time** (e.g. sales *or* traffic) so the story stays easy to follow."
    )
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
    st.info(
        "Data or settings changed since the last run. **Why you see this:** the chart and metrics still show the **last completed run** "
        "so we never mix old results with new choices. Click **Run forecast** in the sidebar to recompute everything together."
    )

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
    st.info(
        "Configure the **sidebar**, then click **Run forecast**. **Why nothing is shown yet:** we only compute after you confirm settings — "
        "that way the page loads quickly and every result block refers to the same run."
    )
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
    st.caption(
        "**Why this section exists:** garbage in → misleading charts out. We report duplicates, bad dates, inferred time step, "
        "and gap handling so you (or a reviewer) can trust what the forecast was fed."
    )
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
    st.caption(
        "**Why a chart *and* numbers:** the line answers ‘where might we go?’; **backtest metrics** answer ‘would this have been credible "
        "on recent history?’ The **baseline** is intentionally simple so you can argue the model adds value."
    )
    if b.get("is_demo"):
        st.caption(DEMO_SCENARIO_CAPTION)
    left, right = st.columns([2, 1])
    with left:
        fig = build_forecast_figure(ts_df, future)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("How to read the chart"):
            st.markdown(
                """
- **Muted line**: observed history after cleaning and resampling. **Why resampled:** so each step is a true “next day/week/month,” not uneven gaps.
- **Accent line**: point forecast (`yhat`) — the model’s best single guess per future period.
- **Shaded band**: **why it’s there** — to show *some* sense of spread around the line. **How we built it:** ±1.96× residual standard deviation from the fit (a common rough “95%” visual). **Caveat:** not a calibrated business guarantee; real drivers we don’t model can push you outside.
- **Odd jumps?** Usually sparse or messy input — cross-check **Input validation**.
                """
            )

    with right:
        st.markdown("**Evaluation (why these metrics)**")
        st.caption(
            "We compare the model to a **naive baseline** (repeat the last value) so ‘lower error’ means more than ‘we drew a pretty curve.’"
        )
        st.caption(f"Backtest holdout (effective): {effective_holdout} {unit}")
        if b["rolling_mode"]:
            st.caption(f"Backtest mode: Rolling ({b['rolling_splits']} splits)")
        if metrics["n_test"] == 0:
            st.write("Not enough data to compute backtest metrics.")
            st.caption(
                "**Why:** the backtest needs a held-out tail plus enough history before it. Short series or aggressive rolling settings can rule that out — better to say ‘no score’ than a fake number."
            )
        elif not np.isfinite(metrics["mape"]) or not np.isfinite(metrics["mae"]):
            st.info(
                "Backtest metrics are not available (non-finite values — often too little data "
                "for the chosen rolling setup)."
            )
            st.caption(
                "**Why:** when a window fails or produces undefined errors, we don’t invent a tidy score; check **Model status** and try fewer rolling splits or a shorter holdout if your series is small."
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
    st.caption(
        "**Why we show this explicitly:** stakeholders should know *which* machinery produced the line — full Holt-Winters, trend-only, or a transparent fallback. Hiding failures would look slick but would be misleading."
    )
    _render_model_status(forecast_meta)

with st.container(border=True):
    st.subheader("Narrative summary")
    st.caption(
        "**Why a text block:** some people understand a short paragraph faster than a chart. **How it’s built:** deterministic rules on your data "
        "(not a black-box LLM) — consistent for demos and easy to reason about."
    )
    st.write(build_explanation(ts_df, future, freq=summary["used_frequency"]))
    st.caption(
        "Useful for walkthroughs, not a substitute for domain review. The chart band reflects residual spread, not a hard bound."
    )

with st.container(border=True):
    st.subheader("Forecast output")
    st.caption(
        "**CSV:** the forecast table for spreadsheets or BI. **JSON:** settings, data health, model path, and metrics for auditors or pipelines. "
        "**Why two formats:** humans vs machines — same run, different consumption."
    )
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
