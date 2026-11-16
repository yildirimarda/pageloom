"""Tests for the pluggable ImageProvider interface and capabilities record."""

import pytest
from PIL import Image

from pageloom.providers import ImageProvider, ImageProviderCapabilities


class FakeProvider:
    """An in-memory fake provider returning solid-color images."""

    capabilities = ImageProviderCapabilities(
        max_reference_images=3,
        supported_sizes=((512, 512), (1024, 1024)),
        supported_aspect_ratios=((1, 1), (16, 9)),
        supports_negative_prompt=True,
        supports_edit=True,
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        references: list[str | Image.Image] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        self.calls.append(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "references": references,
                "negative_prompt": negative_prompt,
                "seed": seed,
            }
        )
        return Image.new("RGB", (width, height), color="blue")


def test_capabilities_record_defaults():
    caps = ImageProviderCapabilities()
    assert caps.max_reference_images == 0
    assert caps.supported_sizes == ()
    assert caps.supported_aspect_ratios == ()
    assert caps.supports_negative_prompt is False
    assert caps.supports_edit is False


def test_capabilities_record_fields():
    caps = ImageProviderCapabilities(
        max_reference_images=2,
        supported_sizes=((512, 512),),
        supported_aspect_ratios=((16, 9),),
        supports_negative_prompt=True,
        supports_edit=False,
    )
    assert caps.max_reference_images == 2
    assert caps.supported_sizes == ((512, 512),)
    assert caps.supported_aspect_ratios == ((16, 9),)
    assert caps.supports_negative_prompt is True
    assert caps.supports_edit is False


def test_capabilities_record_is_immutable():
    caps = ImageProviderCapabilities()
    with pytest.raises(AttributeError):
        caps.max_reference_images = 5  # type: ignore[misc]


def test_fake_provider_is_an_image_provider():
    provider = FakeProvider()
    assert isinstance(provider, ImageProvider)


def test_fake_provider_generate_returns_pil_image_of_requested_size():
    provider = FakeProvider()
    image = provider.generate("a red apple", 512, 512)
    assert isinstance(image, Image.Image)
    assert image.size == (512, 512)
    assert image.mode == "RGB"


def test_fake_provider_passes_through_all_arguments():
    provider = FakeProvider()
    ref_image = Image.new("RGB", (256, 256), color="green")
    provider.generate(
        "a banana",
        width=1024,
        height=1024,
        references=["ref.png", ref_image],
        negative_prompt="blurry",
        seed=42,
    )
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["prompt"] == "a banana"
    assert call["width"] == 1024
    assert call["height"] == 1024
    assert call["references"] == ["ref.png", ref_image]
    assert call["negative_prompt"] == "blurry"
    assert call["seed"] == 42


def test_fake_provider_generate_with_defaults():
    provider = FakeProvider()
    image = provider.generate("a cat", 512, 512)
    assert image.size == (512, 512)
    call = provider.calls[0]
    assert call["references"] is None
    assert call["negative_prompt"] is None
    assert call["seed"] is None


def test_fake_provider_respects_capabilities_max_references():
    provider = FakeProvider()
    assert provider.capabilities.max_reference_images == 3
    assert provider.capabilities.supports_edit is True
    assert provider.capabilities.supports_negative_prompt is True
