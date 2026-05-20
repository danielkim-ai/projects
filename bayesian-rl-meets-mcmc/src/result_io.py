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


def archive_dir(results_dir: Path) -> Path:
    """Return and create the archive directory for timestamped artefacts."""

    archive_path = results_dir / "archive"
    archive_path.mkdir(parents=True, exist_ok=True)
    return archive_path


def environment_archive_dir(results_dir: Path, env_id: str) -> Path:
    """Return the archive directory for a specific MuJoCo environment."""

    env_path = archive_dir(results_dir) / safe_tag(env_id, "environment")
    env_path.mkdir(parents=True, exist_ok=True)
    return env_path


def update_latest_copy(source: Path, latest_name: str) -> Path:
    """Copy the newest result to a latest_* file for downstream consumers."""

    parents = list(source.parents)
    archive_parent = next((parent for parent in parents if parent.name == "archive"), None)
    latest_root = archive_parent.parent if archive_parent is not None else source.parent
    latest_path = latest_root / latest_name
    shutil.copy2(source, latest_path)
    return latest_path
