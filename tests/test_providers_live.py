"""Opt-in live image-generation tests for each pluggable image provider.

Every test is marked ``live`` and skipped unless ``PAGELOOM_LIVE=1`` **and** the
selected provider's credential environment variable is set, so the default test
run stays offline and deterministic. Each live test generates ONE real image
with the real provider (no mocks, no recorded fixtures) and asserts it passes
the quality gate (:mod:`pageloom.quality_gate`).

Providers and their credential env vars (from
:data:`pageloom.providers.IMAGE_PROVIDER_CREDENTIAL_ENV_VARS`):

* ``local``      -- no credential (Stable Diffusion runs locally); gated only
  by ``PAGELOOM_LIVE=1``.
* ``openai``     -- ``OPENAI_API_KEY``
* ``gemini``     -- ``GEMINI_API_KEY``
* ``openrouter`` -- ``OPENROUTER_API_KEY``
* ``flux``       -- any one of ``BFL_API_KEY``, ``FAL_KEY``, ``REPLICATE_API_KEY``
"""

from __future__ import annotations

import os

import pytest
from PIL import Image

from pageloom.providers import (
    IMAGE_PROVIDER_CREDENTIAL_ENV_VARS,
    build_image_provider,
)
from pageloom.quality_gate import assert_passes_quality_gate

PROVIDER_NAMES = ["local", "openai", "gemini", "openrouter", "flux"]

PAGELOOM_LIVE = os.environ.get("PAGELOOM_LIVE") in ("1", "true", "TRUE")


def _provider_credential_set(provider_name: str) -> bool:
    required = IMAGE_PROVIDER_CREDENTIAL_ENV_VARS.get(provider_name, [])
    if not required:
        # ``local`` needs no credential.
        return True
    return any(os.environ.get(var) for var in required)


live = pytest.mark.skipif(
    not PAGELOOM_LIVE,
    reason="set PAGELOOM_LIVE=1 (and the provider's API key) to run live provider tests",
)


@pytest.mark.live
@pytest.mark.skipif(
    not PAGELOOM_LIVE,
    reason="set PAGELOOM_LIVE=1 (and the provider's API key) to run live provider tests",
)
@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
def test_live_provider_generates_one_real_image_passing_quality_gate(provider_name):
    required = IMAGE_PROVIDER_CREDENTIAL_ENV_VARS.get(provider_name, [])
    if required and not any(os.environ.get(var) for var in required):
        pytest.skip(
            f"{provider_name} live test needs PAGELOOM_LIVE=1 and one of {', '.join(required)} set"
        )
    provider = build_image_provider(provider=provider_name)
    if provider.capabilities.supported_sizes:
        width, height = provider.capabilities.supported_sizes[0]
    else:
        width, height = 512, 512
    image = provider.generate("a red apple on a white background", width=width, height=height)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0
    assert_passes_quality_gate(image)
