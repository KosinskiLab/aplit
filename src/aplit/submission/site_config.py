"""Load the active site without coupling service modules to Streamlit."""

from __future__ import annotations

import os
from pathlib import Path

from .models import SiteConfig

SITE_CONFIG_ENV = "APLIT_SITE_CONFIG"


def load_active_site() -> SiteConfig | None:
    raw_path = os.environ.get(SITE_CONFIG_ENV, "").strip()
    if not raw_path:
        return None
    return SiteConfig.load(Path(raw_path).expanduser().resolve())
