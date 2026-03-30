# Forecasting + Explanation App (MVP)

A practical time-series forecasting app for sales and demand data.  
It ingests CSV files, validates and normalizes the series, forecasts future values with uncertainty bands, evaluates performance with backtests, and exports both prediction and metadata artifacts.

## In plain language (for any reader)

**What problem does this solve?**  
You have **one number that changes over time** — for example daily sales, website visits, or orders. People often ask: *“What might the next weeks look like, based on what already happened?”* This app answers that in a structured way: it reads your history, draws a **future continuation** (forecast), and shows **how uncertain** that guess might be (a shaded band — not a guarantee).

**What do I put in?**  
A simple table (CSV): one column for **dates** and one column for the **value** you care about (sales, demand, etc.). Or use the built-in **demo** data to try it without preparing a file.

**What do I get out?**  
- A **chart**: past values + a projected line + a rough “maybe range” around the projection.  
- **Scores** that say how well a similar prediction would have done on **recent** data (so you are not flying blind).  
- A short **text summary** and optional **downloads** (forecast table + a JSON report).

**How do I use it in 30 seconds?**  
Open the app → pick **Demo** or upload CSV → click **Run forecast** in the sidebar → read the chart and the “model status” line.

**Words you might see (mini glossary)**  

| Term | Plain meaning |
|------|----------------|
| **Forecast** | The app’s guess for future dates, based on patterns in the past. |
| **Backtest** | “Pretend we were in the past, predict the next few points, and compare to what really happened.” Helps sanity-check the approach. |
| **Holdout** | How many recent periods we hide during that check — the “exam questions” the model didn’t see while training. |
| **MAPE / MAE** | Error measures: roughly “how far off were we, in %” (MAPE) and “how far off in the same units as your data” (MAE). Lower is better. |
| **Baseline** | A dumb-but-useful reference: “always predict the last known value.” If the app beats that, the model adds some value. |
| **Holt-Winters** | A standard statistical method for series with **trend** (going up/down) and sometimes **seasonality** (e.g. weekdays repeating). |
| **Naive fallback** | If the fancy model fails, the app falls back to “repeat the last value” so you still get a result and a clear warning. |

**What this app is *not***  
It is not a full business planning tool: it does not know your marketing calendar, stockouts, or competitors. It is a **portfolio-grade demo** of a clean pipeline from CSV → validation → forecast → evaluation → export.

## Core Capabilities

- Input validation summary (invalid dates, invalid values, duplicates, inferred frequency)
- Frequency-aware forecasting (`auto`, `D`, `W`, `M`)
- Historical vs forecast visualization with confidence interval
- Backtest metrics (`MAPE`, `MAE`) with naive baseline comparison
- Model status visibility (Holt-Winters vs fallback modes)
- Adaptive holdout defaults by frequency (`D=14`, `W=8`, `M=6`)
- Rolling backtest mode with configurable split count
- Data quality risk warnings for poor input quality
- Strict input contract with explicit, user-friendly error messages
- **Run forecast** action: validation, fit, backtest, and chart run only when you click the sidebar button (not on every widget change)
- In-browser export: download forecast CSV and summary JSON; optional **Save artifacts to project folder** writes `artifacts/forecasts/forecast.csv` and `summary.json`
- Built-in demo datasets (`data/demo_sales`, `data/demo_traffic`): 90 aligned daily rows (2025-01-01 → 2025-03-31). **Sales** uses a hand-tuned 45-day seed (Q1 story: mild post-holiday dip, recovery, then stabilizing growth with weekday seasonality), extended with capped day-to-day dynamics plus small deterministic variation so the tail is not a smooth “rocket.” **Traffic** is derived from sales using the seed’s median ratio and a light weekday factor so both series stay plausible together. Regenerate via `python scripts/generate_demo_ts.py` (see module docstring and `src/demo_scenario.py` for the narrative).
- UI: standard Streamlit layout; dark defaults in `.streamlit/config.toml`; forecast chart colors in `src/visualize.py`.

## Architecture Overview

```text
CSV Input
  -> Column Detection (time/value)
  -> Validation + Cleaning + Resampling
  -> Forecast Engine (Holt-Winters or Naive Fallback)
  -> Backtest Engine (Single Holdout or Rolling)
  -> UI Outputs (chart, metrics, explanation)
  -> Artifacts (forecast.csv, summary.json)
```

## Quick Start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For **tests** (includes `pytest`):

```bash
python -m pip install -r requirements-dev.txt
```

2. Run the app:

```bash
python src/cli.py run
```

3. In the UI: choose **Demo** or upload a CSV, set options, then click **Run forecast**.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `APP_FOOTER_TEXT` | Optional **plain text** shown at the bottom of the Streamlit app and embedded in **exported summary JSON** when set. Empty or unset = no footer. |

Local: copy `.env.example` to `.env` and set values (loaded via `python-dotenv` when installed).

## CLI Commands

| Purpose | Command |
|------|-------|
| Run app | `python src/cli.py run` |
| Run tests | `python src/cli.py test` |
| Run end-to-end smoke check | `python src/cli.py smoke` |
| Generate demo sample outputs | `python src/cli.py generate` |
| Regenerate bundled demo CSVs (seed + rule-based extension) | `python scripts/generate_demo_ts.py` |

After changing demo inputs, refresh committed sample forecasts with `python src/cli.py generate`.

## Input Contract

Minimum requirements:

- One datetime-like column (for example: `date`, `datetime`, `timestamp`)
- One numeric target column (for example: `value`, `sales`, `demand`, `visitors`)
- At least 10 unique time points (30+ recommended for better stability)

Supported frequency modes:

- `auto`: inferred from median time gaps
- `D`: daily
- `W`: weekly
- `M`: monthly

Typical validation checks:

- Invalid date parsing count
- Invalid numeric value count
- Duplicate timestamps
- Missing periods after resampling

## Forecasting & Fallback Policy

Primary model:

- Holt-Winters with **additive trend** and optional **additive seasonality**, chosen from the resampled frequency and history length:
  - **Daily (`D`)**: 7-day season only if there are at least 28 observations; otherwise trend-only Holt-Winters
  - **Weekly (`W`)**: 52-week season only if there are at least 104 observations; otherwise trend-only
  - **Monthly (`M`)**: 12-month season only if there are at least 36 observations; otherwise trend-only

Statuses:

- `holt_winters_seasonal`: seasonal term enabled
- `holt_winters_trend_only`: credible fit without seasonal component (short history or non-daily frequency)
- `naive_short_history`: history shorter than the minimum for Holt-Winters in this app
- `naive_fallback`: Holt-Winters did not complete; last-value forecast. `fallback_reason` distinguishes import issues vs invalid input vs numerical failure vs other (`statsmodels_import_error`, `fit_invalid_input`, `fit_numerical_error`, `model_error`). `fit_error_message` carries a truncated exception string when available.

Model status, seasonality label, and fit diagnostics are shown in the UI and in exported JSON (`hw_seasonality`, `fit_error_message`, `resample`, `backtest_fit_*`).

## Backtesting Policy

Two backtest modes are available:

- Single holdout backtest
- Rolling backtest (multiple consecutive splits, averaged metrics)

Primary metrics:

- `MAPE` (percentage error, scale-independent)
- `MAE` (absolute error magnitude, scale-dependent)

Baseline:

- Naive last-value baseline is always computed for comparison.

## Output Artifacts

### `artifacts/forecasts/forecast.csv`

Forecast table with:

- `date`
- `yhat`
- `yhat_lower`
- `yhat_upper`

### `artifacts/forecasts/summary.json`

Metadata report including:

- configuration (`horizon`, `holdout_effective`, `rolling_mode`, `rolling_splits`, `frequency`)
- input summary (`rows_input`, `rows_prepared`, date range)
- model info (`model_status`, `fallback_reason`, diagnostics when applicable)
- metrics (`mape`, `mae`, baseline metrics, `n_test`, optional `splits_used`)
- optional `app_footer` when `APP_FOOTER_TEXT` is set in the environment

## Limitations

- This MVP is univariate (single target series).
- External drivers (promotions, holidays, pricing, campaigns) are not modeled explicitly.
- Confidence intervals represent uncertainty, not guaranteed bounds.
- Very short or low-quality series can reduce forecast reliability.

## FAQ

**Why are metrics missing?**  
Your series may be too short for the selected holdout/split setup.

**Why did the app switch to naive fallback?**  
Model fitting failed or history is insufficient. Check model status in the UI and `summary.json`.

**Why does the forecast look flat?**  
This commonly happens with short/noisy history or fallback mode.

**Why does monthly forecasting use month-start dates?**  
Monthly mode is normalized to month-start indexing for consistent period stepping.
