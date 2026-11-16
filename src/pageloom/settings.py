"""Environment-variable overrides for PageLoom settings."""

from __future__ import annotations

import os


def get_model_id() -> str:
    """Default Stable Diffusion model id.

    The model id is configuration: it is read from settings/env and never
    hardcoded beyond this default. Override with ``PAGELOOM_MODEL_ID``.
    """
    return os.environ.get("PAGELOOM_MODEL_ID", "runwayml/stable-diffusion-v1-5")


def get_profile() -> str:
    """Generation profile: ``fast`` (fewer inference steps, quick previews) or
    ``quality`` (production defaults: 25-30 steps, guidance ~7).

    Override with the ``PAGELOOM_PROFILE`` environment variable.
    """
    return os.environ.get("PAGELOOM_PROFILE", "quality")


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


def get_quality_gate_max_retries() -> int:
    """Max cheap-heuristic retries with a fresh seed before a frame is rejected.

    Each rejected frame is regenerated once with a new seed, up to this many
    retries. Set ``PAGELOOM_QUALITY_GATE_MAX_RETRIES=0`` to disable retries
    (the first bad frame is rejected immediately).
    """
    raw = os.environ.get("PAGELOOM_QUALITY_GATE_MAX_RETRIES")
    if raw is None or raw.strip() == "":
        return 2
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def get_image_model() -> str | None:
    """Explicit image model id, or ``None`` to let the provider pick its default.

    Override with the ``PAGELOOM_IMAGE_MODEL`` environment variable.
    """
    return os.environ.get("PAGELOOM_IMAGE_MODEL") or None


def get_image_strength() -> float:
    """Image-to-image strength for local Stable Diffusion reference conditioning.

    ``0.0`` (the default) means off: plain text-to-image generation and the
    example page is not used as a reference. Any value in ``(0, 1]`` enables
    image-to-image generation with the example page as the reference image;
    higher values make the output follow the reference more closely.

    Override with the ``PAGELOOM_IMAGE_STRENGTH`` environment variable. Values
    outside ``[0, 1]`` are clamped.
    """
    raw = os.environ.get("PAGELOOM_IMAGE_STRENGTH")
    if raw is None or raw.strip() == "":
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, value))


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


def get_num_inference_steps() -> int:
    """Production default number of diffusion inference steps (25 to 30).

    These settings are what make a local Stable Diffusion run capable of
    producing a real, usable image rather than noise. Override with the
    ``PAGELOOM_SD_STEPS`` environment variable. The ``fast`` profile defaults
    to 6 steps for quick previews; ``quality`` defaults to 25.
    """
    raw = os.environ.get("PAGELOOM_SD_STEPS")
    if raw is None or raw.strip() == "":
        if get_profile() == "fast":
            return 6
        return 25
    try:
        return max(1, int(raw))
    except ValueError:
        return 6 if get_profile() == "fast" else 25


def get_guidance_scale() -> float:
    """Production default classifier-free guidance scale (around 7).

    Override with the ``PAGELOOM_SD_GUIDANCE_SCALE`` environment variable.
    The ``fast`` profile defaults to 5.0; ``quality`` defaults to 7.5.
    """
    raw = os.environ.get("PAGELOOM_SD_GUIDANCE_SCALE")
    if raw is None or raw.strip() == "":
        if get_profile() == "fast":
            return 5.0
        return 7.5
    try:
        return float(raw)
    except ValueError:
        return 5.0 if get_profile() == "fast" else 7.5


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


_DEFAULT_IMAGE_PRICE_TABLE: dict[str, dict[str, float]] = {
    # Local Stable Diffusion runs on the user's own hardware: no per-image fee.
    "local": {"default": 0.0},
    # OpenAI Images API (gpt-image-2 / dall-e-3 family), per image.
    "openai": {"gpt-image-2": 0.040, "dall-e-3": 0.080, "default": 0.040},
    # Google Gemini Nano Banana family, per image.
    "gemini": {"gemini-nano-banana-2.1": 0.030, "default": 0.030},
    # OpenRouter image models: free auto-router costs nothing; paid models vary.
    "openrouter": {"openrouter/free": 0.0, "openrouter/auto": 0.050, "default": 0.050},
    # Black Forest Labs FLUX.2, per image.
    "flux": {"flux-pro-1.1": 0.055, "default": 0.055},
}


def get_image_price_table() -> dict[str, dict[str, float]]:
    """Configurable price table: estimated USD cost per generated image.

    Returns a mapping ``{provider_name: {model_id: cost_usd, "default": ...}}``.
    The table can be overridden wholesale with the ``PAGELOOM_IMAGE_PRICE_TABLE``
    environment variable, which accepts either a JSON object or a path to a JSON
    file containing one. Model-specific prices fall back to the entry's
    ``"default"`` key, then to ``0.0`` when the model is unknown.
    """
    raw = os.environ.get("PAGELOOM_IMAGE_PRICE_TABLE")
    if raw:
        import json
        from pathlib import Path

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            price_path = Path(raw)
            if price_path.is_file():
                return json.loads(price_path.read_text(encoding="utf-8"))
    return _DEFAULT_IMAGE_PRICE_TABLE


def image_price_usd(provider_name: str, model_id: str | None = None) -> float:
    """Estimated USD cost for one image from ``provider_name`` / ``model_id``.

    Looks the price up in the configurable price table (see
    :func:`get_image_price_table`). Returns ``0.0`` for providers not in the
    table (e.g. local generation) or for unknown models.
    """
    table = get_image_price_table()
    entry = table.get(provider_name)
    if entry is None:
        return 0.0
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        if model_id and model_id in entry:
            return float(entry[model_id])
        if "default" in entry:
            return float(entry["default"])
    return 0.0
