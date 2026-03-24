from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.smoke import run_demo_smoke


def test_demo_sales_csv_matches_seed_anchor() -> None:
    """Guards accidental demo regen without updating the documented Q1 scenario."""
    base_dir = Path(__file__).resolve().parents[1]
    sales_csv = base_dir / "data" / "demo_sales" / "sales_ts.csv"
    assert sales_csv.is_file()
    first = int(pd.read_csv(sales_csv).iloc[0]["sales"])
    assert first == 112


def test_run_demo_smoke_outputs_created() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    report = run_demo_smoke(base_dir=base_dir, horizon=8, holdout=7)

    assert "sales" in report
    assert "traffic" in report

    for key in ("sales", "traffic"):
        section = report[key]
        assert section["rows_in"] > 0
        assert section["rows_out"] == 8
        assert section["backtest_n"] in (0, 7)
        assert "forecast_model_status" in section
        assert Path(section["output_path"]).exists()
        assert Path(section["summary_path"]).exists()

