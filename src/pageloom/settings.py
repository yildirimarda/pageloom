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


def get_image_provider() -> str:
    """Image-generation provider: ``local`` (default) or a registered provider.

    Override with the ``PAGELOOM_IMAGE_PROVIDER`` environment variable.
    """
    return os.environ.get("PAGELOOM_IMAGE_PROVIDER", "local")


def get_image_model() -> str | None:
    """Explicit image model id, or ``None`` to let the provider pick its default.

    Override with the ``PAGELOOM_IMAGE_MODEL`` environment variable.
    """
    return os.environ.get("PAGELOOM_IMAGE_MODEL") or None


def get_openai_image_model() -> str:
    """OpenAI image-generation model id.

    Override with the ``OPENAI_IMAGE_MODEL`` environment variable.
    """
    return os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")


def get_gemini_image_model() -> str:
    """Google Gemini Nano Banana image model id.

    Override with the ``GEMINI_IMAGE_MODEL`` environment variable.
    """
    return os.environ.get("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")


def get_openrouter_image_model() -> str:
    """OpenRouter image-generation model id.

    Override with the ``OPENROUTER_IMAGE_MODEL`` environment variable.
    Defaults to OpenRouter's auto-router, which selects a model that supports
    the requested capabilities (here image output) without hardcoding a
    specific free model id, since the model roster changes over time.
    """
    return os.environ.get("OPENROUTER_IMAGE_MODEL", "openrouter/auto")


def get_flux_host() -> str:
    """FLUX host: ``bfl`` (Black Forest Labs' own API), ``fal`` or ``replicate``.

    Override with the ``PAGELOOM_FLUX_HOST`` environment variable.
    """
    return os.environ.get("PAGELOOM_FLUX_HOST", "bfl")


def get_flux_model() -> str:
    """Black Forest Labs FLUX.2 model id.

    Override with the ``PAGELOOM_FLUX_MODEL`` environment variable.
    Defaults to ``flux-pro-1.1``.
    """
    return os.environ.get("PAGELOOM_FLUX_MODEL", "flux-pro-1.1")


def get_bfl_api_key() -> str | None:
    """Credential for Black Forest Labs' own API.

    Override with ``BFL_API_KEY`` or ``PAGELOOM_BFL_API_KEY``.
    """
    return os.environ.get("BFL_API_KEY") or os.environ.get("PAGELOOM_BFL_API_KEY") or None


def get_fal_api_key() -> str | None:
    """Credential for fal.ai.

    Override with ``FAL_KEY`` or ``PAGELOOM_FAL_KEY``.
    """
    return os.environ.get("FAL_KEY") or os.environ.get("PAGELOOM_FAL_KEY") or None


def get_replicate_api_key() -> str | None:
    """Credential for Replicate.

    Override with ``REPLICATE_API_KEY`` or ``PAGELOOM_REPLICATE_API_KEY``.
    """
    return (
        os.environ.get("REPLICATE_API_KEY") or os.environ.get("PAGELOOM_REPLICATE_API_KEY") or None
    )
