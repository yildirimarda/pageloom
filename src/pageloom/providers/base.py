"""Provider-agnostic interface for image-generation backends.

Defines the `ImageProvider` protocol every image backend (local Stable
Diffusion, OpenAI gpt-image, Google Nano Banana, Black Forest Labs FLUX.2,
OpenRouter image models, ...) must satisfy, plus an `ImageProviderCapabilities`
record describing what a provider can do so callers (e.g. `Pipeline`) can
negotiate sizes, reference images, negative prompts and edits without knowing
the concrete provider.

Model IDs are configuration (see docs/IMAGE-BACKENDS.md): they are read from
settings/env, never hardcoded here beyond a settings value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class ImageProviderCapabilities:
    """What an image provider can do.

    ``supported_sizes`` lists the exact (width, height) pixel sizes the
    provider accepts natively (empty means "any size"). ``supported_aspect_ratios``
    lists aspect ratios as (width, height) tuples when the provider only
    supports fixed ratios (empty means "any ratio"). Populate at least one of
    the two for a useful provider.
    """

    max_reference_images: int = 0
    supported_sizes: tuple[tuple[int, int], ...] = ()
    supported_aspect_ratios: tuple[tuple[int, int], ...] = ()
    supports_negative_prompt: bool = False
    supports_edit: bool = False


@runtime_checkable
class ImageProvider(Protocol):
    """Provider-agnostic interface for image-generation APIs.

    Concrete providers implement `generate` and expose `capabilities`. Callers
    depend only on this protocol, never on a concrete provider class.
    """

    capabilities: ImageProviderCapabilities

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        references: list[str | Image.Image] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image: ...

    def estimated_cost_usd(self, width: int, height: int) -> float: ...
