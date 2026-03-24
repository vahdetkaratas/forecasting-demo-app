from __future__ import annotations

from src.run_fingerprints import (
    data_fingerprint_demo,
    data_fingerprint_upload_id,
    data_fingerprint_upload_none,
    data_fingerprint_upload_sha256,
    settings_fingerprint,
)


def test_demo_fingerprint_distinguishes_series() -> None:
    assert data_fingerprint_demo("Sales demo") != data_fingerprint_demo("Traffic demo")


def test_upload_sha256_stable() -> None:
    fp = data_fingerprint_upload_sha256("abc123")
    assert fp == ("upload", "sha256", "abc123")


def test_settings_fingerprint_tracks_all_controls() -> None:
    a = settings_fingerprint(30, 14, True, True, 3, "auto")
    b = settings_fingerprint(31, 14, True, True, 3, "auto")
    c = settings_fingerprint(30, 14, False, True, 3, "auto")
    assert a != b
    assert a != c


def test_upload_id_includes_size() -> None:
    assert data_fingerprint_upload_id("x", 100) != data_fingerprint_upload_id("x", 101)


def test_upload_none_singleton() -> None:
    assert data_fingerprint_upload_none() == ("upload", "none")
