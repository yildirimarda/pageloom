"""Cost awareness: per-image price table, Pipeline cost summation, --max-cost,
--estimate (Milestone 9 item)."""

import json
from unittest.mock import patch

import pytest
from PIL import Image

from pageloom import settings
from pageloom.cli import main
from pageloom.pipeline import Pipeline
from pageloom.providers import (
    IMAGE_PROVIDER_CREDENTIAL_ENV_VARS,
    IMAGE_PROVIDER_MAP,
    BFLImageProvider,
    GeminiImageProvider,
    ImageProvider,
    ImageProviderCapabilities,
    LocalDiffusersProvider,
    OpenAIImageProvider,
    OpenRouterImageProvider,
)


class _CostProvider:
    """Fake provider with a fixed per-image cost and a call log."""

    provider_name = "demo"
    capabilities = ImageProviderCapabilities()

    def __init__(
        self,
        cost_usd: float = 0.05,
        model_id: str | None = None,
        verify_license: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.cost_usd = cost_usd
        self.model_id = model_id
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        references: list | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        self.calls.append({"prompt": prompt, "width": width, "height": height})
        return Image.new("RGB", (width, height), color="red")

    def estimated_cost_usd(self, width: int, height: int) -> float:
        return self.cost_usd


# -- price table -----------------------------------------------------------


def test_price_table_defaults_include_local_free():
    table = settings.get_image_price_table()
    assert table["local"]["default"] == 0.0


def test_image_price_usd_unknown_provider_is_zero():
    assert settings.image_price_usd("no-such-provider") == 0.0


def test_image_price_usd_model_specific_price(monkeypatch):
    monkeypatch.setenv(
        "PAGELOOM_IMAGE_PRICE_TABLE",
        json.dumps({"openai": {"gpt-image-2": 0.040, "default": 0.090}}),
    )
    assert settings.image_price_usd("openai", "gpt-image-2") == 0.040


def test_image_price_usd_unknown_model_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(
        "PAGELOOM_IMAGE_PRICE_TABLE",
        json.dumps({"openai": {"gpt-image-2": 0.040, "default": 0.090}}),
    )
    assert settings.image_price_usd("openai", "dall-e-3") == 0.090


def test_image_price_usd_env_var_as_file_path(tmp_path, monkeypatch):
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps({"flux": {"flux-pro-1.1": 0.123}}), encoding="utf-8")
    monkeypatch.setenv("PAGELOOM_IMAGE_PRICE_TABLE", str(price_path))
    assert settings.image_price_usd("flux", "flux-pro-1.1") == 0.123


# -- per-provider estimated cost ------------------------------------------


def test_local_provider_estimated_cost_is_zero(tmp_path):
    with (
        patch("pageloom.providers.local_diffusers.StableDiffusionPipeline.from_pretrained"),
        patch("pageloom.providers.local_diffusers.torch.cuda.is_available", return_value=False),
        patch(
            "pageloom.providers.local_diffusers.torch.backends.mps.is_available",
            return_value=False,
        ),
    ):
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = lambda device: mock_pipeline
        provider = LocalDiffusersProvider(verify_license=False)
        assert provider.estimated_cost_usd(512, 512) == 0.0


def test_openai_estimated_cost_from_table(monkeypatch):
    monkeypatch.setenv(
        "PAGELOOM_IMAGE_PRICE_TABLE",
        json.dumps({"openai": {"gpt-image-2": 0.040, "default": 0.090}}),
    )
    provider = OpenAIImageProvider()
    assert provider.estimated_cost_usd(1024, 1024) == 0.040
    provider = OpenAIImageProvider(model_id="unknown-model")
    assert provider.estimated_cost_usd(1024, 1024) == 0.090


def test_gemini_estimated_cost_from_table(monkeypatch):
    monkeypatch.setenv(
        "PAGELOOM_IMAGE_PRICE_TABLE",
        json.dumps({"gemini": {"gemini-nano-banana-2.1": 0.030, "default": 0.050}}),
    )
    provider = GeminiImageProvider(model_id="gemini-nano-banana-2.1")
    assert provider.estimated_cost_usd(1024, 1024) == 0.030
    provider = GeminiImageProvider(model_id="unknown-model")
    assert provider.estimated_cost_usd(1024, 1024) == 0.050


def test_openrouter_estimated_cost_from_table(monkeypatch):
    monkeypatch.setenv(
        "PAGELOOM_IMAGE_PRICE_TABLE",
        json.dumps(
            {"openrouter": {"openrouter/free": 0.0, "openrouter/auto": 0.050, "default": 0.050}}
        ),
    )
    provider = OpenRouterImageProvider(model_id="openrouter/free")
    assert provider.estimated_cost_usd(1024, 1024) == 0.0
    provider = OpenRouterImageProvider(model_id="openrouter/auto")
    assert provider.estimated_cost_usd(1024, 1024) == 0.050


def test_flux_estimated_cost_from_table(monkeypatch):
    monkeypatch.setenv(
        "PAGELOOM_IMAGE_PRICE_TABLE",
        json.dumps({"flux": {"flux-pro-1.1": 0.055, "default": 0.070}}),
    )
    provider = BFLImageProvider(model_id="flux-pro-1.1")
    assert provider.estimated_cost_usd(1024, 1024) == 0.055
    provider = BFLImageProvider(model_id="unknown-model")
    assert provider.estimated_cost_usd(1024, 1024) == 0.070


def test_estimated_cost_usd_is_part_of_image_provider_protocol():
    provider = _CostProvider(cost_usd=0.025)
    assert isinstance(provider, ImageProvider)
    assert provider.estimated_cost_usd(1024, 1024) == 0.025


# -- Pipeline cost summation ----------------------------------------------


def _make_pipeline(tmp_path, generator, page_structure=None, max_cost_usd=None):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="blue").save(img_path)
    return Pipeline(
        example_image=str(img_path),
        generator=generator,
        page_structure=page_structure,
        max_cost_usd=max_cost_usd,
    )


def test_pipeline_estimated_total_cost_single_label(tmp_path):
    provider = _CostProvider(cost_usd=0.05)
    pipeline = _make_pipeline(tmp_path, provider)
    # single_letter_word_illustration: 1 image per page, $0.05 each
    assert pipeline.estimated_total_cost_usd(4) == 4 * 0.05
    # Estimating must not call the provider or compile anything.
    assert provider.calls == []
    assert pipeline.total_cost_usd == 0.0


def test_pipeline_estimated_total_cost_multiple_labels(tmp_path):
    provider = _CostProvider(cost_usd=0.05)
    pipeline = _make_pipeline(
        tmp_path,
        provider,
        page_structure={
            "content_pattern": "multi_item_scene",
            "element_count": 3,
            "arrangement": "grid",
            "text_image_relationship": "multiple_labels",
        },
    )
    # 2 pages x 3 sub-images x $0.05 = $0.30
    assert pipeline.estimated_total_cost_usd(2) == pytest.approx(2 * 3 * 0.05)
    assert provider.calls == []


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_max_cost_stops_before_exceeding(mock_compile, tmp_path, capsys):
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _CostProvider(cost_usd=0.05)
    pipeline = _make_pipeline(tmp_path, provider, max_cost_usd=0.12)
    # $0.12 budget affords 2 images ($0.10); the 3rd ($0.15) would exceed it.
    pipeline.run(contents=["a", "b", "c", "d"], output_path=str(tmp_path / "out.pdf"))
    assert pipeline.budget_stopped is True
    assert pipeline.total_cost_usd == pytest.approx(0.10)
    assert len(provider.calls) == 2  # only two provider calls
    pages = mock_compile.call_args.args[0]
    assert len(pages) == 2
    err = capsys.readouterr().err
    assert "max cost" in err


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_max_cost_none_generates_all(mock_compile, tmp_path):
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _CostProvider(cost_usd=0.05)
    pipeline = _make_pipeline(tmp_path, provider, max_cost_usd=None)
    pipeline.run(contents=["a", "b", "c"], output_path=str(tmp_path / "out.pdf"))
    assert pipeline.budget_stopped is False
    assert pipeline.total_cost_usd == pytest.approx(3 * 0.05)
    assert len(provider.calls) == 3


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_max_cost_cached_images_are_free(mock_compile, tmp_path):
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _CostProvider(cost_usd=0.05)
    pipeline = _make_pipeline(tmp_path, provider, max_cost_usd=0.04)
    # First run: $0.04 budget affords exactly one image ($0.05 > $0.04? no:
    # 0 + 0.05 = 0.05 > 0.04, so even the first image is unaffordable -> 0 pages.
    pipeline.run(contents=["a"], output_path=str(tmp_path / "out.pdf"))
    assert pipeline.budget_stopped is True
    assert len(provider.calls) == 0


# -- CLI --estimate -------------------------------------------------------


def test_cli_estimate_prints_cost_without_calling_provider(tmp_path, capsys, monkeypatch):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="blue").save(img_path)
    provider = _CostProvider(cost_usd=0.07)
    monkeypatch.setitem(IMAGE_PROVIDER_MAP, "demo", _CostProvider)
    monkeypatch.setitem(IMAGE_PROVIDER_CREDENTIAL_ENV_VARS, "demo", [])
    monkeypatch.setenv("PAGELOOM_IMAGE_PROVIDER", "demo")
    with patch("pageloom.pipeline.build_image_provider", return_value=provider):
        code = main(
            [
                "generate",
                "--example",
                str(img_path),
                "--pages",
                "3",
                "--no-vision",
                "--image-provider",
                "demo",
                "--estimate",
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
    assert code == 0
    out = capsys.readouterr().out
    assert "Estimated cost:" in out
    assert "$0.21" in out  # 3 pages * $0.07
    assert provider.calls == []  # no provider call during estimate


def test_cli_estimate_local_provider_is_free(tmp_path, capsys, monkeypatch):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="blue").save(img_path)
    monkeypatch.setenv("PAGELOOM_IMAGE_PROVIDER", "local")
    with (
        patch("pageloom.providers.local_diffusers.StableDiffusionPipeline.from_pretrained"),
        patch("pageloom.providers.local_diffusers.torch.cuda.is_available", return_value=False),
        patch(
            "pageloom.providers.local_diffusers.torch.backends.mps.is_available",
            return_value=False,
        ),
    ):
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = lambda device: mock_pipeline
        code = main(
            [
                "generate",
                "--example",
                str(img_path),
                "--pages",
                "3",
                "--no-vision",
                "--estimate",
            ]
        )
    assert code == 0
    assert "$0.00" in capsys.readouterr().out


# -- CLI --max-cost -------------------------------------------------------


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_cli_max_cost_stops_run(mock_compile, tmp_path, capsys, monkeypatch):
    mock_compile.return_value = tmp_path / "out.pdf"
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="blue").save(img_path)
    provider = _CostProvider(cost_usd=0.05)
    monkeypatch.setitem(IMAGE_PROVIDER_MAP, "demo", _CostProvider)
    monkeypatch.setitem(IMAGE_PROVIDER_CREDENTIAL_ENV_VARS, "demo", [])
    with patch("pageloom.pipeline.build_image_provider", return_value=provider):
        code = main(
            [
                "generate",
                "--example",
                str(img_path),
                "--pages",
                "5",
                "--no-vision",
                "--image-provider",
                "demo",
                "--max-cost",
                "0.12",
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
    assert code == 0
    # Budget affords 2 images; run stops before exceeding, still exits 0.
    assert len(provider.calls) == 2
    pages = mock_compile.call_args.args[0]
    assert len(pages) == 2
    assert "max cost" in capsys.readouterr().err
