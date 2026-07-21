from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
AKS_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"
DATABASE_PATH = DATA_DIR / "app.db"
FEATURE_MODELS_PATH = PROJECT_DIR / "config" / "feature_models.json"


def ensure_data_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_profiles() -> dict[str, dict[str, Any]]:
    if not FEATURE_MODELS_PATH.exists():
        return {}
    raw = json.loads(FEATURE_MODELS_PATH.read_text(encoding="utf-8"))
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("feature_models.json: profiles must be an object")
    return {str(key): dict(value) for key, value in profiles.items()}

