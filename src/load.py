from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(file_or_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_or_path)

