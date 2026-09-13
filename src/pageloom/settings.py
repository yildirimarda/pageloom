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


def get_vision_provider() -> str:
    """Default provider used for automatic visual page-structure analysis.

    Defaults to OpenRouter so a free, zero-cost vision-capable model can be
    used out of the box (OpenRouter's `openrouter/free` auto-router selects
    a free model that supports image understanding). Override with
    PAGELOOM_VISION_PROVIDER if you'd rather use OpenAI or Anthropic.
    """
    return os.environ.get("PAGELOOM_VISION_PROVIDER") or os.environ.get(
        "PAGELOOM_LLM_PROVIDER", "openrouter"
    )


def get_openai_vision_model() -> str:
    return os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")


def get_anthropic_vision_model() -> str:
    return os.environ.get("ANTHROPIC_VISION_MODEL", "claude-3-haiku-20240307")


def get_openrouter_vision_model() -> str:
    # "openrouter/free" is OpenRouter's auto-router: it selects a free model
    # at zero cost and filters for the capabilities the request needs (here,
    # image understanding). This avoids hardcoding a specific free model ID,
    # since the free model roster changes over time.
    return os.environ.get("OPENROUTER_VISION_MODEL", "openrouter/free")
