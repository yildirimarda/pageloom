"""Environment-variable overrides for PageLoom settings."""

from __future__ import annotations

import os


def get_model_id() -> str:
    return os.environ.get("PAGELOOM_MODEL_ID", "runwayml/stable-diffusion-v1-5")


def get_device() -> str | None:
    return os.environ.get("PAGELOOM_DEVICE") or None


def get_output_dir() -> str | None:
    return os.environ.get("PAGELOOM_OUTPUT_DIR") or None


def get_hf_cache_dir() -> str | None:
    return os.environ.get("PAGELOOM_HF_CACHE_DIR") or os.environ.get("HF_HOME") or None
