"""Pure fingerprints for Streamlit run state (unit-testable, no Streamlit imports)."""

from __future__ import annotations


def data_fingerprint_demo(demo_name: str) -> tuple[object, ...]:
    return ("demo", demo_name or "")


def data_fingerprint_upload_none() -> tuple[object, ...]:
    return ("upload", "none")


def data_fingerprint_upload_id(file_id: str, size: int) -> tuple[object, ...]:
    return ("upload", "id", file_id, size)


def data_fingerprint_upload_sha256(digest_hex: str) -> tuple[object, ...]:
    return ("upload", "sha256", digest_hex)


def settings_fingerprint(
    horizon: int,
    holdout: int,
    adaptive_holdout: bool,
    rolling_mode: bool,
    rolling_splits: int,
    freq_mode: str,
) -> tuple[object, ...]:
    return (horizon, holdout, adaptive_holdout, rolling_mode, rolling_splits, freq_mode)
