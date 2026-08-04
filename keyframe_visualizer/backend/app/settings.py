from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
AKS_ROOT = PROJECT_DIR.parent
VSI_ROOT = AKS_ROOT / "VSI_VideoFraming"
VSI_BUNDLED_OUTPUT_DIR = VSI_ROOT / "output"
VSI_BUNDLED_EASYOCR_DIR = VSI_BUNDLED_OUTPUT_DIR / "easyocr_models"
VSI_BUNDLED_YOLO_MODEL = VSI_ROOT / "yolov8s-worldv2.pt"
VSI_BUNDLED_CLIP_MODEL = VSI_ROOT / "weights" / "clip" / "ViT-B-32.pt"
VSI_BUNDLED_TEXT_MODEL = (
    VSI_ROOT
    / "weights"
    / "sentence_transformer"
    / "paraphrase-multilingual-mpnet-base-v2"
)
DATA_DIR = PROJECT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"
SESSIONS_DIR = DATA_DIR / "sessions"
TRASH_DIR = DATA_DIR / ".trash"
DATABASE_PATH = DATA_DIR / "app.db"
CLIP_MODELS_DIR = DATA_DIR / "models" / "clip"
VSI_MODEL_CACHE_DIR = DATA_DIR / "models" / "vsi"
FEATURE_MODELS_PATH = PROJECT_DIR / "config" / "feature_models.json"
VLM_MODELS_PATH = PROJECT_DIR / "config" / "vlm_models.json"
ENV_PATH = PROJECT_DIR / ".env"

load_dotenv(ENV_PATH, override=False)


def ensure_data_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VSI_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_profiles() -> dict[str, dict[str, Any]]:
    if not FEATURE_MODELS_PATH.exists():
        return {}
    raw = json.loads(FEATURE_MODELS_PATH.read_text(encoding="utf-8"))
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("feature_models.json: profiles must be an object")
    return {str(key): dict(value) for key, value in profiles.items()}


def feature_profile_status() -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for profile_id, profile in load_feature_profiles().items():
        config = profile.get("config", {})
        required = sorted(
            str(value)
            for key, value in config.items()
            if str(key).endswith("_env") and value
        )
        missing = [name for name in required if not os.getenv(name)]
        statuses[profile_id] = {
            "enabled": bool(profile.get("enabled", True)),
            "credentials_ready": not missing,
            "required_environment_variables": required,
            "missing_environment_variables": missing,
        }
    return statuses


def load_vlm_profiles() -> dict[str, dict[str, Any]]:
    if not VLM_MODELS_PATH.exists():
        return {}
    raw = json.loads(VLM_MODELS_PATH.read_text(encoding="utf-8"))
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("vlm_models.json: profiles must be an object")
    return {str(key): dict(value) for key, value in profiles.items()}


def vlm_profile_status() -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for profile_id, profile in load_vlm_profiles().items():
        config = profile.get("config", {})
        required = sorted(
            str(value)
            for key, value in config.items()
            if str(key).endswith("_env") and value
        )
        missing = [name for name in required if not os.getenv(name)]
        statuses[profile_id] = {
            "id": profile_id,
            "name": str(profile.get("name", profile_id)),
            "backend": str(profile.get("backend", "mep")),
            "enabled": bool(profile.get("enabled", True)),
            "credentials_ready": not missing,
            "required_environment_variables": required,
            "missing_environment_variables": missing,
            "max_frames": int(config.get("max_frames", 32)),
        }
    return statuses
