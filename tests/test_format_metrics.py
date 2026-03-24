from __future__ import annotations

import math

import pytest

from src.format_metrics import format_finite_metric


def test_format_finite_metric_formats_number() -> None:
    assert format_finite_metric(3.14159) == "3.14"


def test_format_finite_metric_non_finite_becomes_dash() -> None:
    assert format_finite_metric(float("nan")) == "—"
    assert format_finite_metric(float("inf")) == "—"


def test_format_finite_metric_custom_pattern() -> None:
    assert format_finite_metric(2.5, pattern="{:.1f}") == "2.5"


def test_format_finite_metric_int() -> None:
    assert format_finite_metric(7) == "7.00"


@pytest.mark.parametrize("bad", [None, "x", object()])
def test_format_finite_metric_rejects_non_numeric(bad: object) -> None:
    assert format_finite_metric(bad) == "—"  # type: ignore[arg-type]
