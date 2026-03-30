"""Read optional app footer from environment (plain text, e.g. portfolio branding)."""

from __future__ import annotations

import os


def app_footer_text() -> str | None:
    raw = os.environ.get("APP_FOOTER_TEXT", "")
    text = raw.strip()
    return text if text else None
