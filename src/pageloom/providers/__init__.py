"""Pluggable image-generation providers."""

from __future__ import annotations

import os

from pageloom import settings
from pageloom.providers.base import ImageProvider, ImageProviderCapabilities
from pageloom.providers.gemini import GeminiImageProvider
from pageloom.providers.local_diffusers import LocalDiffusersProvider
from pageloom.providers.openai import OpenAIImageProvider
from pageloom.providers.openrouter import (
    OpenRouterImageProvider,
    list_openrouter_image_models,
)

IMAGE_PROVIDER_MAP = {
    "local": LocalDiffusersProvider,
    "openai": OpenAIImageProvider,
    "gemini": GeminiImageProvider,
    "openrouter": OpenRouterImageProvider,
}

# Environment variables (in preference order) that supply the credential for
# each provider. Providers that need no credentials (currently ``local``) map
# to an empty list.
IMAGE_PROVIDER_CREDENTIAL_ENV_VARS: dict[str, list[str]] = {
    "local": [],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def get_image_provider_class(name: str):
    """Return the provider class registered for ``name``."""
    provider_cls = IMAGE_PROVIDER_MAP.get(name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown image provider '{name}'. Supported: {', '.join(IMAGE_PROVIDER_MAP)}"
        )
    return provider_cls


def build_image_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> ImageProvider:
    """Build the image provider selected by settings or an explicit argument.

    ``provider`` defaults to :func:`pageloom.settings.get_image_provider`
    (``local`` unless ``PAGELOOM_IMAGE_PROVIDER`` is set). ``model`` defaults
    to :func:`pageloom.settings.get_image_model` (``None`` lets the provider
    pick its own default).

    Raises :class:`RuntimeError` naming the missing credential variable when a
    credential-requiring provider is selected without its key and no
    ``api_key`` is passed explicitly.
    """
    provider_name = provider or settings.get_image_provider()
    provider_cls = get_image_provider_class(provider_name)

    required_env = IMAGE_PROVIDER_CREDENTIAL_ENV_VARS.get(provider_name, [])
    if required_env and not api_key:
        missing = [var for var in required_env if not os.environ.get(var)]
        if missing:
            raise RuntimeError(
                f"Image provider '{provider_name}' selected but credential is missing. "
                f"Set the appropriate environment variable ({', '.join(missing)}) "
                f"or pass api_key explicitly."
            )

    return provider_cls(model_id=model, verify_license=False)


__all__ = [
    "IMAGE_PROVIDER_CREDENTIAL_ENV_VARS",
    "IMAGE_PROVIDER_MAP",
    "GeminiImageProvider",
    "ImageProvider",
    "ImageProviderCapabilities",
    "LocalDiffusersProvider",
    "OpenAIImageProvider",
    "OpenRouterImageProvider",
    "build_image_provider",
    "get_image_provider_class",
    "list_openrouter_image_models",
]
