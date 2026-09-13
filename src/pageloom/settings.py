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


def get_openai_llm_model() -> str:
    return os.environ.get("PAGELOOM_LLM_MODEL", "gpt-4o-mini")


def get_anthropic_llm_model() -> str:
    return os.environ.get("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307")


def get_openrouter_llm_model() -> str:
    return os.environ.get("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
