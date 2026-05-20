"""Result naming utilities that prevent experiment artefact overwrites."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


def safe_tag(tag: str | None, fallback: str) -> str:
    """Return a filesystem-safe experiment tag."""

    raw = tag or fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    return cleaned.strip("._") or fallback


def run_id(save_tag: str | None, phase: str, seed: int) -> str:
    """Create a stable, human-readable run identifier."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_tag(save_tag, phase)}_seed{seed}_{timestamp}"


def update_latest_copy(source: Path, latest_name: str) -> Path:
    """Copy the newest result to a latest_* file for downstream consumers."""

    latest_path = source.with_name(latest_name)
    shutil.copy2(source, latest_path)
    return latest_path

