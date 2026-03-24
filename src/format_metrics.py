"""Display helpers for numeric metrics (Streamlit-free)."""

from __future__ import annotations

import numpy as np


def format_finite_metric(x: float, pattern: str = "{:.2f}") -> str:
    return pattern.format(x) if isinstance(x, (int, float)) and np.isfinite(x) else "—"
