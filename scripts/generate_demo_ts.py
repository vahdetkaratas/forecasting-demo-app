#!/usr/bin/env python3
"""
Regenerate bundled demo daily CSVs.

Scenario (portfolio-facing): synthetic Q1 **e-commerce demand** — early January with normal
weekday variation, a **mild mid-month dip** (post-holiday softness), **recovery** through late
January, then **stabilizing growth** into March. Not meant to mimic real noise; meant to read
clearly in a demo while still looking like a plausible business series.

Mechanics:
- First 45 rows: hand-tuned seed (2025-01-01 .. 2025-02-14).
- Sales extension: day-to-day ratios from the seed, capped multipliers, then deterministic
  wiggles so the tail is not a smooth exponential.
- Traffic extension: rebuilt from sales using the seed’s median sales→visitors ratio plus
  a mild weekday factor so the two demos stay correlated and plausible.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Hand-tuned canonical seed (do not edit without updating README / expectations).
START_DATE = "2025-01-01"
SEED_DAYS = 45
EXTRA_DAYS = 45

SALES_SEED: list[int] = [
    112,
    118,
    121,
    106,
    103,
    119,
    124,
    126,
    129,
    124,
    110,
    107,
    116,
    118,
    128,
    132,
    129,
    118,
    115,
    135,
    141,
    143,
    147,
    144,
    130,
    127,
    152,
    158,
    160,
    164,
    162,
    148,
    145,
    170,
    175,
    178,
    182,
    180,
    165,
    162,
    188,
    192,
    195,
    200,
    198,
]

VISITORS_SEED: list[int] = [
    770,
    811,
    831,
    729,
    708,
    818,
    852,
    865,
    886,
    852,
    756,
    736,
    797,
    811,
    879,
    906,
    886,
    811,
    790,
    927,
    968,
    981,
    1009,
    988,
    893,
    872,
    1043,
    1084,
    1097,
    1124,
    1111,
    1015,
    995,
    1165,
    1200,
    1220,
    1247,
    1234,
    1131,
    1111,
    1288,
    1315,
    1336,
    1370,
    1356,
]


def _transition_ratios(dates: pd.DatetimeIndex, values: np.ndarray) -> dict[tuple[int, int], float]:
    dow = dates.dayofweek.to_numpy(dtype=int)
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i in range(1, len(values)):
        prev, cur = int(dow[i - 1]), int(dow[i])
        buckets[(prev, cur)].append(float(values[i]) / max(float(values[i - 1]), 1.0))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def _extend_from_seed(
    seed_values: list[int],
    extra_days: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if len(seed_values) != SEED_DAYS:
        raise ValueError(f"Seed length must be {SEED_DAYS}, got {len(seed_values)}")

    dates = pd.date_range(START_DATE, periods=SEED_DAYS, freq="D")
    values = np.array(seed_values, dtype=float)
    ratios = _transition_ratios(dates, values)
    mean_step = float(np.mean(np.diff(values[-14:])))

    last_v = float(values[-1])
    seed_last = last_v
    prev_dow = int(dates.dayofweek[-1])
    out = list(values.tolist())
    cur = dates[-1]
    med = float(np.median(values))
    fallback_mult = 1.0 + (mean_step / med if med > 0 else 0.015)

    # Cap multipliers so the extended tail does not explode (portfolio charts looked unreal).
    cap_mult = 1.028
    tight_mult = 1.014
    scale_trigger = 1.55

    for _ in range(extra_days):
        cur = cur + pd.Timedelta(days=1)
        dow = int(cur.dayofweek)
        key = (prev_dow, dow)
        mult = min(ratios.get(key, fallback_mult), cap_mult)
        if last_v > seed_last * scale_trigger:
            mult = min(mult, tight_mult)
        last_v = max(1.0, round(last_v * mult))
        out.append(last_v)
        prev_dow = dow

    # Deterministic micro-variation (not i.i.d. noise — reproducible repo builds).
    for i in range(SEED_DAYS, len(out)):
        bump = round(4.0 * math.sin(0.27 * i) + 3.0 * math.cos(0.19 * (i + 2)))
        out[i] = max(1, int(out[i] + bump))

    full_dates = pd.date_range(START_DATE, periods=len(out), freq="D")
    return full_dates, np.array(out, dtype=int)


def _align_visitors_to_sales(sales: np.ndarray, visitors_seed: list[int]) -> np.ndarray:
    """Keep seed visitors fixed; rebuild extended visitors from sales × stable ratio + weekday lift."""
    if len(visitors_seed) != SEED_DAYS:
        raise ValueError("Visitors seed length mismatch")
    out: list[int] = list(visitors_seed)
    ratio_med = float(
        np.median(
            np.array(visitors_seed, dtype=float)
            / np.maximum(sales[:SEED_DAYS].astype(float), 1.0)
        )
    )
    for i in range(SEED_DAYS, len(sales)):
        wf = 0.93 + 0.14 * ((i % 7) / 7.0)
        out.append(max(1, int(round(float(sales[i]) * ratio_med * wf))))
    return np.array(out, dtype=int)


def write_series(path: Path, dates: pd.DatetimeIndex, col_name: str, values: np.ndarray) -> None:
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), col_name: values})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write demo_sales and demo_traffic daily CSVs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (parent of scripts/).",
    )
    parser.add_argument(
        "--extra",
        type=int,
        default=EXTRA_DAYS,
        help=f"Days to append after the {SEED_DAYS}-day seed (default: {EXTRA_DAYS}).",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    sales_path = root / "data" / "demo_sales" / "sales_ts.csv"
    traffic_path = root / "data" / "demo_traffic" / "traffic_ts.csv"

    s_dates, s_vals = _extend_from_seed(SALES_SEED, args.extra)
    t_vals = _align_visitors_to_sales(s_vals, VISITORS_SEED)
    t_dates = s_dates

    assert len(s_dates) == SEED_DAYS + args.extra
    assert (s_dates == t_dates).all()
    assert np.array_equal(s_vals[:SEED_DAYS], np.array(SALES_SEED))
    assert np.array_equal(t_vals[:SEED_DAYS], np.array(VISITORS_SEED))

    write_series(sales_path, s_dates, "sales", s_vals)
    write_series(traffic_path, t_dates, "visitors", t_vals)

    print(f"Wrote {len(s_vals)} rows -> {sales_path}")
    print(f"Wrote {len(t_vals)} rows -> {traffic_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
