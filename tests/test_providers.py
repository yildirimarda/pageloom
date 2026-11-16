"""Tests for the pluggable ImageProvider interface and capabilities record."""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pageloom import settings
from pageloom.providers import (
    IMAGE_PROVIDER_CREDENTIAL_ENV_VARS,
    IMAGE_PROVIDER_MAP,
    GeminiImageProvider,
    ImageProvider,
    ImageProviderCapabilities,
    LocalDiffusersProvider,
    OpenAIImageProvider,
    OpenRouterImageProvider,
    build_image_provider,
    list_openrouter_image_models,
)
from pageloom.providers.size_negotiation import (
    fit_to_slot,
    generate_fitting_slot,
    negotiate_size,
)


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

    def estimated_cost_usd(self, width: int, height: int) -> float:
        return 0.0


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


@patch("pageloom.providers.local_diffusers.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.providers.local_diffusers.torch.cuda.is_available", return_value=False)
@patch("pageloom.providers.local_diffusers.torch.backends.mps.is_available", return_value=False)
def test_local_diffusers_provider_is_an_image_provider(mock_mps, mock_cuda, mock_from_pretrained):
    """LocalDiffusersProvider must satisfy the ImageProvider protocol."""
    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_from_pretrained.return_value = mock_pipeline

    provider = LocalDiffusersProvider(verify_license=False)
    assert isinstance(provider, ImageProvider)
    assert provider.capabilities.max_reference_images == 0
    assert provider.capabilities.supports_negative_prompt is True
    assert provider.capabilities.supports_edit is False


@patch("pageloom.providers.local_diffusers.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.providers.local_diffusers.torch.cuda.is_available", return_value=False)
@patch("pageloom.providers.local_diffusers.torch.backends.mps.is_available", return_value=False)
def test_local_diffusers_provider_generate_conforms_to_interface(
    mock_mps, mock_cuda, mock_from_pretrained
):
    """generate() takes the ImageProvider arguments and returns a PIL image."""
    from PIL import Image

    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_image = Image.new("RGB", (512, 512), color="red")
    mock_pipeline.return_value = MagicMock(images=[mock_image])
    mock_from_pretrained.return_value = mock_pipeline

    provider = LocalDiffusersProvider(verify_license=False)
    result = provider.generate(
        "a red apple",
        width=512,
        height=512,
        references=["ref.png"],
        negative_prompt="blurry",
        seed=7,
    )
    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)

    kwargs = mock_pipeline.call_args.kwargs
    assert kwargs["prompt"] == "a red apple"
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["width"] == 512
    assert kwargs["height"] == 512
    assert kwargs["generator"] is not None


@contextmanager
def _mock_sd_backend():
    """Context manager stubbing out the real Stable Diffusion load."""
    with (
        patch(
            "pageloom.providers.local_diffusers.StableDiffusionPipeline.from_pretrained"
        ) as mock_from,
        patch("pageloom.providers.local_diffusers.torch.cuda.is_available", return_value=False),
        patch(
            "pageloom.providers.local_diffusers.torch.backends.mps.is_available", return_value=False
        ),
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = lambda device: mock_pipeline
        mock_from.return_value = mock_pipeline
        yield mock_from, mock_pipeline


def test_build_image_provider_defaults_to_local(monkeypatch):
    monkeypatch.setenv("PAGELOOM_IMAGE_PROVIDER", "local")
    with _mock_sd_backend() as (mock_from, _mock_pipeline):
        provider = build_image_provider()
        assert isinstance(provider, LocalDiffusersProvider)
        assert provider.model_id == "runwayml/stable-diffusion-v1-5"
        mock_from.assert_called_once()


def test_build_image_provider_explicit_provider_and_model(monkeypatch):
    monkeypatch.setenv("PAGELOOM_IMAGE_PROVIDER", "local")
    with _mock_sd_backend() as (mock_from, _mock_pipeline):
        provider = build_image_provider(provider="local", model="custom/image-model")
        assert isinstance(provider, LocalDiffusersProvider)
        assert provider.model_id == "custom/image-model"
        call_args = mock_from.call_args
        assert call_args.args[0] == "custom/image-model"


def test_build_image_provider_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown image provider 'not-a-real-provider'"):
        build_image_provider(provider="not-a-real-provider")


def test_build_image_provider_missing_credential_raises_clear_error(monkeypatch):
    class DemoProvider:
        capabilities = ImageProviderCapabilities()

        def generate(
            self,
            prompt: str,
            width: int,
            height: int,
            references: list | None = None,
            negative_prompt: str | None = None,
            seed: int | None = None,
        ):
            return Image.new("RGB", (width, height))

    monkeypatch.setitem(IMAGE_PROVIDER_MAP, "demo", DemoProvider)
    monkeypatch.setitem(IMAGE_PROVIDER_CREDENTIAL_ENV_VARS, "demo", ["PAGELOOM_DEMO_KEY"])
    with (
        patch.dict("os.environ", {"PAGELOOM_IMAGE_PROVIDER": "demo"}, clear=True),
        pytest.raises(RuntimeError, match="PAGELOOM_DEMO_KEY"),
    ):
        build_image_provider()


class _FixedSizeProvider:
    """Provider that only generates at its native size and records calls."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.calls: list[tuple[int, int]] = []

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        references: list[str | Image.Image] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        self.calls.append((width, height))
        return Image.new("RGB", self.size, color="red")

    def estimated_cost_usd(self, width: int, height: int) -> float:
        return 0.0


# -- negotiate_size: capability profiles with square, tall and wide targets --


def test_negotiate_size_passthrough_when_provider_accepts_any_size():
    caps = ImageProviderCapabilities()
    assert negotiate_size(612, 792, caps) == (612, 792)
    assert negotiate_size(100, 100, None) == (100, 100)


def test_negotiate_size_square_target_picks_nearest_supported_size():
    caps = ImageProviderCapabilities(supported_sizes=((512, 512), (1024, 1024)))
    assert negotiate_size(612, 612, caps) == (512, 512)
    assert negotiate_size(400, 400, caps) == (512, 512)
    assert negotiate_size(900, 900, caps) == (1024, 1024)


def test_negotiate_size_tall_and_wide_targets_map_to_nearest_supported_size():
    caps = ImageProviderCapabilities(supported_sizes=((512, 512), (1024, 1024)))
    # Tall target (512x768) and wide target (768x512) both have area closest
    # to 512x512 among the supported sizes.
    assert negotiate_size(512, 768, caps) == (512, 512)
    assert negotiate_size(768, 512, caps) == (512, 512)


def test_negotiate_size_nearest_ratio_scales_to_fit_within_request():
    caps = ImageProviderCapabilities(supported_aspect_ratios=((1, 1), (16, 9)))
    assert negotiate_size(600, 600, caps) == (600, 600)  # square target
    # Tall target (2:3) is closest to square -> largest square that fits.
    assert negotiate_size(512, 768, caps) == (512, 512)
    # Wide target (3:2) is closest to 16:9 -> 768x432 fits within 768x512.
    assert negotiate_size(768, 512, caps) == (768, 432)
    # Exact 16:9 request maps losslessly.
    assert negotiate_size(800, 450, caps) == (800, 450)


def test_negotiate_size_supported_sizes_take_precedence_over_ratios():
    caps = ImageProviderCapabilities(
        supported_sizes=((512, 512),), supported_aspect_ratios=((16, 9),)
    )
    assert negotiate_size(768, 512, caps) == (512, 512)


# -- fit_to_slot: crop or pad back to the layout slot --


def test_fit_to_slot_exact_size_is_no_op():
    img = Image.new("RGB", (400, 300), color="red")
    assert fit_to_slot(img, 400, 300) is img


def test_fit_to_slot_crops_larger_image_to_center():
    img = Image.new("RGB", (600, 400), color="blue")
    out = fit_to_slot(img, 400, 300)
    assert out.size == (400, 300)
    # Center crop: top-left of output is (100, 50) of the source.
    assert out.getpixel((0, 0)) == img.getpixel((100, 50))


def test_fit_to_slot_pads_smaller_image_with_white():
    img = Image.new("RGB", (300, 200), color="green")
    out = fit_to_slot(img, 400, 300)
    assert out.size == (400, 300)
    assert out.getpixel((0, 0)) == (255, 255, 255)  # padding is white
    assert out.getpixel((50, 50)) == (0, 128, 0)  # green content centered


def test_fit_to_slot_crops_one_dim_and_pads_the_other():
    img = Image.new("RGB", (800, 200), color="red")
    out = fit_to_slot(img, 400, 600)
    assert out.size == (400, 600)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((200, 300)) == (255, 0, 0)


def test_fit_to_slot_passes_through_non_pil_images():
    obj = object()
    assert fit_to_slot(obj, 10, 10) is obj


# -- generate_fitting_slot: negotiate, generate, fit --


def test_generate_fitting_slot_uses_provider_native_size_and_fits_to_slot():
    provider = FakeProvider()
    # FakeProvider supports (512, 512) and (1024, 1024); 612x792 is closest to
    # 512x512 by area, so generation happens there and the result is padded
    # back to the 612x792 layout slot.
    image = generate_fitting_slot(provider, "a cat", 612, 792)
    assert image.size == (612, 792)
    call = provider.calls[0]
    assert call["width"] == 512
    assert call["height"] == 512


def test_generate_fitting_slot_crops_when_generated_is_larger():
    provider = FakeProvider()
    image = generate_fitting_slot(provider, "a cat", 300, 200)
    assert image.size == (300, 200)
    call = provider.calls[0]
    assert call["width"] == 512
    assert call["height"] == 512


def test_generate_fitting_slot_with_aspect_ratios_only():
    provider = _FixedSizeProvider((480, 270))
    provider.capabilities = ImageProviderCapabilities(
        supported_aspect_ratios=((16, 9),)
    )  # 480x270 is exactly 16:9
    image = generate_fitting_slot(provider, "a cat", 480, 480)
    assert image.size == (480, 480)
    # Largest 16:9 size fitting in 480x480 is 480x270, then padded to square.
    assert provider.calls[0] == (480, 270)
    assert image.getpixel((0, 0)) == (255, 255, 255)  # top/bottom padding


# -- OpenAI image provider --

_VALID_B64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
    "DwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@contextmanager
def _mock_openai_images(b64_json=_VALID_B64_PNG):
    """Patch sys.modules['openai'] with a fake client for image generation."""
    import sys

    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_data = MagicMock()
    mock_data.b64_json = b64_json
    mock_response = MagicMock(data=[mock_data])
    mock_client.images.generate.return_value = mock_response
    mock_client.images.edit.return_value = mock_response
    mock_module.OpenAI.return_value = mock_client

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_module
    try:
        yield mock_client
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_openai_image_provider_is_an_image_provider():
    provider = OpenAIImageProvider()
    assert isinstance(provider, ImageProvider)
    assert provider.capabilities.max_reference_images == 1
    assert provider.capabilities.supports_edit is True
    assert provider.capabilities.supported_sizes == ((1024, 1024),)


def test_openai_image_provider_model_id_read_from_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    provider = OpenAIImageProvider()
    assert provider.model_id == "gpt-image-1"


def test_openai_image_provider_model_id_defaults_to_settings_default(monkeypatch):
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    provider = OpenAIImageProvider()
    assert provider.model_id == settings.get_openai_image_model()


def test_openai_image_provider_generate_calls_images_generate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with _mock_openai_images() as mock_client:
        provider = OpenAIImageProvider(api_key="test-key")
        image = provider.generate("a red apple", width=1024, height=1024)
        assert isinstance(image, Image.Image)
        call_kwargs = mock_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == settings.get_openai_image_model()
        assert call_kwargs["prompt"] == "a red apple"
        assert call_kwargs["n"] == 1
        assert call_kwargs["size"] == "1024x1024"
        mock_client.images.edit.assert_not_called()


def test_openai_image_provider_generate_with_reference_calls_images_edit(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with _mock_openai_images() as mock_client:
        provider = OpenAIImageProvider(api_key="test-key")
        ref = Image.new("RGB", (1024, 1024), color="red")
        image = provider.generate(
            "make it match the example", width=1024, height=1024, references=[ref]
        )
        assert isinstance(image, Image.Image)
        call_kwargs = mock_client.images.edit.call_args.kwargs
        assert call_kwargs["model"] == settings.get_openai_image_model()
        assert call_kwargs["prompt"] == "make it match the example"
        assert call_kwargs["n"] == 1
        assert call_kwargs["size"] == "1024x1024"
        assert call_kwargs["image"] is not None
        mock_client.images.generate.assert_not_called()


def test_openai_image_provider_generate_with_reference_file_path(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (1024, 1024), color="green").save(ref_path)
    with _mock_openai_images() as mock_client:
        provider = OpenAIImageProvider(api_key="test-key")
        image = provider.generate("match this", width=1024, height=1024, references=[str(ref_path)])
        assert isinstance(image, Image.Image)
        call_kwargs = mock_client.images.edit.call_args.kwargs
        assert call_kwargs["image"] == str(ref_path)


def test_openai_image_provider_missing_credential_raises_clear_error(monkeypatch):
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
    ):
        build_image_provider(provider="openai")


@contextmanager
def _mock_gemini_genai():
    """Patch sys.modules['google.genai'] with a fake client for image generation."""
    import base64
    import sys

    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data.data = base64.b64decode(_VALID_B64_PNG)
    mock_content = MagicMock(parts=[mock_part])
    mock_candidate = MagicMock(content=mock_content)
    mock_response = MagicMock(candidates=[mock_candidate])
    mock_client.models.generate_content.return_value = mock_response
    mock_module.Client.return_value = mock_client

    original_modules = dict(sys.modules)
    sys.modules["google.genai"] = mock_module
    try:
        yield mock_client
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_gemini_image_provider_is_an_image_provider():
    provider = GeminiImageProvider()
    assert isinstance(provider, ImageProvider)
    assert provider.capabilities.max_reference_images == 1
    assert provider.capabilities.supports_edit is True
    assert provider.capabilities.supported_sizes == ((1024, 1024),)


def test_gemini_image_provider_model_id_read_from_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
    provider = GeminiImageProvider()
    assert provider.model_id == "gemini-3.1-flash-image"


def test_gemini_image_provider_model_id_defaults_to_settings_default(monkeypatch):
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    provider = GeminiImageProvider()
    assert provider.model_id == settings.get_gemini_image_model()


def test_gemini_image_provider_generate_with_references(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
    with _mock_gemini_genai() as mock_client:
        provider = GeminiImageProvider(api_key="test-key")
        ref = Image.new("RGB", (1024, 1024), color="red")
        image = provider.generate(
            "make it match the example",
            width=1024,
            height=1024,
            references=[ref],
        )
        assert isinstance(image, Image.Image)
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-nano-banana-2.1"
        contents = call_kwargs["contents"]
        assert contents[0] == "make it match the example"
        assert contents[1] is ref  # reference image passed as a request part


def test_gemini_image_provider_generate_without_references(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with _mock_gemini_genai() as mock_client:
        provider = GeminiImageProvider(api_key="test-key")
        image = provider.generate("a red apple", width=1024, height=1024)
        assert isinstance(image, Image.Image)
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == settings.get_gemini_image_model()
        assert call_kwargs["contents"] == ["a red apple"]


def test_gemini_image_provider_generate_with_reference_file_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (1024, 1024), color="green").save(ref_path)
    with _mock_gemini_genai() as mock_client:
        provider = GeminiImageProvider(api_key="test-key")
        image = provider.generate("match this", width=1024, height=1024, references=[str(ref_path)])
        assert isinstance(image, Image.Image)
        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert isinstance(contents[1], Image.Image)


def test_gemini_image_provider_missing_credential_raises_clear_error(monkeypatch):
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="GEMINI_API_KEY"),
    ):
        build_image_provider(provider="gemini")


# -- OpenRouter image provider --


@contextmanager
def _mock_openrouter_chat(b64_json=_VALID_B64_PNG):
    """Patch sys.modules['openai'] with a fake client for OpenRouter image gen."""
    import sys

    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_image = MagicMock()
    mock_image.b64_json = b64_json
    mock_message = MagicMock(images=[mock_image])
    mock_choice = MagicMock(message=mock_message)
    mock_response = MagicMock(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_response
    mock_module.OpenAI.return_value = mock_client

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_module
    try:
        yield mock_client
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_openrouter_image_provider_is_an_image_provider():
    provider = OpenRouterImageProvider()
    assert isinstance(provider, ImageProvider)
    assert provider.capabilities.max_reference_images == 1
    assert provider.capabilities.supports_edit is False
    assert provider.capabilities.supported_sizes == ((1024, 1024),)


def test_openrouter_image_provider_model_id_read_from_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "openai/gpt-image-2")
    provider = OpenRouterImageProvider()
    assert provider.model_id == "openai/gpt-image-2"


def test_openrouter_image_provider_model_id_defaults_to_settings_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_IMAGE_MODEL", raising=False)
    provider = OpenRouterImageProvider()
    assert provider.model_id == settings.get_openrouter_image_model()


def test_openrouter_image_provider_generate_sends_modalities_image(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with _mock_openrouter_chat() as mock_client:
        provider = OpenRouterImageProvider(api_key="test-key")
        image = provider.generate("a red apple", width=1024, height=1024)
        assert isinstance(image, Image.Image)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == settings.get_openrouter_image_model()
        assert call_kwargs["messages"] == [{"role": "user", "content": "a red apple"}]
        assert call_kwargs["extra_body"]["modalities"] == ["image"]
        assert call_kwargs["extra_body"]["width"] == 1024
        assert call_kwargs["extra_body"]["height"] == 1024


def test_openrouter_image_provider_generate_returns_image_from_b64_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with _mock_openrouter_chat() as _mock_client:
        provider = OpenRouterImageProvider(api_key="test-key")
        image = provider.generate("a banana", width=512, height=512)
        assert isinstance(image, Image.Image)
        assert image.size == (1, 1)


def test_openrouter_image_provider_missing_credential_raises_clear_error(monkeypatch):
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"),
    ):
        build_image_provider(provider="openrouter")


def test_list_openrouter_image_models_filters_by_output_modality(monkeypatch):
    payload = {
        "data": [
            {"id": "openai/gpt-image-2", "architecture": {"modality": "text->image"}},
            {"id": "black-forest-labs/flux-pro", "architecture": {"modality": "text->image"}},
            {"id": "google/gemma-4-26b-a4b-it", "architecture": {"modality": "text->text"}},
            {"id": "openrouter/free", "architecture": {"modality": "text->image"}},
        ]
    }

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    with patch(
        "pageloom.providers.openrouter.urllib.request.urlopen",
        return_value=_FakeResponse(),
    ):
        models = list_openrouter_image_models()

    assert "openai/gpt-image-2" in models
    assert "black-forest-labs/flux-pro" in models
    assert "openrouter/free" in models
    assert "google/gemma-4-26b-a4b-it" not in models
