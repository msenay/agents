#!/usr/bin/env python3
"""
Environment utilities for demos and scripts.

Automatically load .env from common locations without overriding existing envs:
 - Project root (auto-detected)
 - Current working directory
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _find_project_root(start: Path | None = None) -> Path:
    """Best-effort project root discovery by looking for common markers."""
    start = start or Path(__file__).resolve()
    current = start if start.is_dir() else start.parent
    markers = {".git", "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"}

    while True:
        if any((current / m).exists() for m in markers):
            return current
        if current.parent == current:
            # filesystem root reached; fallback to two levels up from this file
            return Path(__file__).resolve().parents[2]
        current = current.parent


def load_env_if_exists() -> None:
    """Load environment variables from .env if present.

    Search order (override=False):
      1) ProjectRoot/.env
      2) CWD/.env
      3) Fallback: load_dotenv() default search
    """
    if load_dotenv is None:
        return

    # 1) Project root
    project_root = _find_project_root()
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=False)

    # 2) CWD
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=str(cwd_env), override=False)

    # 3) Default heuristic (only if neither above loaded anything extra)
    # This won't override already-set env vars due to override=False default in dotenv
    load_dotenv(override=False)


