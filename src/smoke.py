from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from .forecast import forecast_series, run_backtest
    from .footer_config import app_footer_text
    from .load import load_csv
    from .resample import prepare_series
    from .time_detect import detect_time_column, detect_value_column
except ImportError:
    from forecast import forecast_series, run_backtest
    from footer_config import app_footer_text
    from load import load_csv
    from resample import prepare_series
    from time_detect import detect_time_column, detect_value_column


def _metric_for_json(x: float) -> float | None:
    xf = float(x)
    return round(xf, 4) if math.isfinite(xf) else None


def run_demo_smoke(base_dir: Path, horizon: int = 14, holdout: int = 14) -> dict:
    """
    Run an end-to-end smoke check on bundled demo datasets.

    Returns a compact report with generated file paths and row counts.
    """
    data_dir = base_dir / "data"
    output_dir = data_dir / "sample_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    demos = {
        "sales": data_dir / "demo_sales" / "sales_ts.csv",
        "traffic": data_dir / "demo_traffic" / "traffic_ts.csv",
    }

    for name, file_path in demos.items():
        raw_df = load_csv(file_path)
        date_col = detect_time_column(raw_df)
        value_col = detect_value_column(raw_df, date_col)
        ts_df, summary = prepare_series(raw_df, date_col, value_col, freq="auto")
        fc, fc_meta = forecast_series(ts_df, horizon=horizon, confidence=0.95, freq=summary["used_frequency"])
        metrics = run_backtest(ts_df, holdout=holdout, freq=summary["used_frequency"])

        out_file = output_dir / f"forecast_{name}_sample.csv"
        fc_out = fc.copy()
        for col in ("yhat", "yhat_lower", "yhat_upper"):
            if col in fc_out.columns:
                fc_out[col] = fc_out[col].round(4)
        fc_out.to_csv(out_file, index=False)
        summary_file = output_dir / f"forecast_{name}_summary.json"
        payload = {
            "rows_in": int(len(ts_df)),
            "rows_out": int(len(fc)),
            "frequency": summary["used_frequency"],
            "forecast_model_status": fc_meta["model_status"],
            "backtest": {
                "mape": _metric_for_json(metrics["mape"]) if metrics["n_test"] else None,
                "mae": _metric_for_json(metrics["mae"]) if metrics["n_test"] else None,
                "baseline_mape": _metric_for_json(metrics["baseline_mape"]) if metrics["n_test"] else None,
                "baseline_mae": _metric_for_json(metrics["baseline_mae"]) if metrics["n_test"] else None,
                "n_test": int(metrics["n_test"]),
            },
        }
        footer = app_footer_text()
        if footer:
            payload["app_footer"] = footer
        summary_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        report[name] = {
            "rows_in": int(len(ts_df)),
            "rows_out": int(len(fc)),
            "output_path": str(out_file),
            "summary_path": str(summary_file),
            "frequency": summary["used_frequency"],
            "backtest_n": int(metrics["n_test"]),
            "forecast_model_status": fc_meta["model_status"],
        }

    return report

