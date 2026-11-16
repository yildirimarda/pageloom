"""Tests for the automatic image quality gate (Milestone 10).

Covers the cheap deterministic heuristics (near-black, near-uniform, tiled,
noise-like) and the pipeline wiring: retry with a fresh seed up to a limit,
the optional vision-model subject check, and reporting a rejected frame as a
failure instead of writing it into the PDF.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pageloom.pipeline import Pipeline
from pageloom.providers.base import ImageProviderCapabilities
from pageloom.quality_gate import (
    QualityGateError,
    assert_passes_quality_gate,
    is_black,
    is_noise_like,
    is_tiled,
    is_uniform,
    quality_gate_reasons,
)


def _gradient(width: int, height: int) -> Image.Image:
    """A non-uniform illustration that passes the quality gate."""
    img = Image.new("L", (width, height))
    img.putdata([int(255 * x / max(1, width - 1)) for y in range(height) for x in range(width)])
    return img.convert("RGB")


def _black(width: int = 512, height: int = 512) -> Image.Image:
    return Image.new("RGB", (width, height), color="black")


def _make_pipeline(tmp_path: Path, generator, **kwargs) -> Pipeline:
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="green").save(img_path)
    kwargs.setdefault("verify_consistency", False)
    return Pipeline(example_image=str(img_path), generator=generator, **kwargs)


class _ScriptedProvider:
    """Returns pre-scripted images and records every generate() call."""

    capabilities = ImageProviderCapabilities()

    def __init__(self, images: list[Image.Image]) -> None:
        self.images = list(images)
        self.calls: list[dict] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        references: list | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        self.calls.append({"prompt": prompt, "width": width, "height": height, "seed": seed})
        if self.images:
            return self.images.pop(0)
        return _black(width, height)

    def estimated_cost_usd(self, width: int, height: int) -> float:
        return 0.0


# -- cheap heuristics ---------------------------------------------------------


def test_is_black_detects_near_black_frame():
    assert is_black(_black())
    assert is_black(Image.new("RGB", (32, 32), color=(4, 2, 6)))
    assert not is_black(_gradient(64, 64))
    assert not is_black(Image.new("RGB", (32, 32), color=(9, 9, 9)))


def test_is_uniform_detects_blank_frame():
    assert is_uniform(Image.new("RGB", (32, 32), color="white"))
    assert is_uniform(_black())
    assert is_uniform(Image.new("RGB", (32, 32), color=(12, 12, 12)))
    assert not is_uniform(_gradient(64, 64))


def test_is_tiled_detects_repeated_tile():
    # A multi-colour tile repeated across the frame: not uniform, so is_uniform
    # does not catch it, but the repeated block pattern does.
    tile = Image.new("RGB", (32, 32))
    tile.putdata(
        [(255, 0, 0)] * 256 + [(0, 255, 0)] * 256 + [(0, 0, 255)] * 256 + [(255, 255, 0)] * 256
    )
    tiled = Image.new("RGB", (64, 64))
    for y in range(2):
        for x in range(2):
            tiled.paste(tile, (x * 32, y * 32))
    assert is_tiled(tiled)
    assert not is_tiled(_gradient(64, 64))
    assert not is_tiled(_black())


def test_is_noise_like_detects_random_noise():
    rng = __import__("random").Random(7)
    noise = Image.new("RGB", (128, 128))
    noise.putdata(
        [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(128 * 128)]
    )
    assert is_noise_like(noise)
    assert not is_noise_like(_gradient(128, 128))
    assert not is_noise_like(_black())
    assert not is_noise_like(Image.new("RGB", (128, 128), color="white"))


def test_quality_gate_reports_each_failing_heuristic():
    assert quality_gate_reasons(_black()) == ["near-black frame", "near-uniform frame"]
    assert quality_gate_reasons(Image.new("RGB", (32, 32), color="white")) == ["near-uniform frame"]
    assert quality_gate_reasons(_gradient(64, 64)) == []


def test_assert_passes_quality_gate_raises_on_failure():
    with pytest.raises(AssertionError, match="quality gate"):
        assert_passes_quality_gate(_black())
    # A passing image raises nothing.
    assert_passes_quality_gate(_gradient(64, 64))


def test_quality_gate_error_carries_reasons():
    exc = QualityGateError(["near-black frame", "noise-like frame"], image=_black())
    assert exc.reasons == ["near-black frame", "noise-like frame"]
    assert "near-black frame" in str(exc)


# -- pipeline wiring: retry and failure reporting -----------------------------


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_rejects_black_frame_after_retries(mock_compile, tmp_path: Path):
    """A near-black frame is retried with a fresh seed up to the limit, then
    reported as a failure: QualityGateError is raised and nothing is compiled."""
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _ScriptedProvider([_black() for _ in range(5)])
    pipeline = _make_pipeline(tmp_path, provider)

    with pytest.raises(QualityGateError) as exc:
        pipeline.run(contents=["apple"], output_path=str(tmp_path / "out.pdf"))

    assert "near-black frame" in exc.value.reasons
    # 1 initial attempt + 2 retries (default PAGELOOM_QUALITY_GATE_MAX_RETRIES).
    assert provider.call_count == 3
    mock_compile.assert_not_called()


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_retries_then_accepts_good_frame(mock_compile, tmp_path: Path):
    """A frame that fails once but passes on a later seed is accepted."""
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _ScriptedProvider([_black(), _black(), _gradient(512, 512)])
    pipeline = _make_pipeline(tmp_path, provider)

    result = pipeline.run(contents=["apple"], output_path=str(tmp_path / "out.pdf"))

    assert isinstance(result, Path)
    assert provider.call_count == 3
    mock_compile.assert_called_once()


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_max_retries_zero_rejects_immediately(mock_compile, tmp_path: Path):
    """PAGELOOM_QUALITY_GATE_MAX_RETRIES=0 rejects the first bad frame."""
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _ScriptedProvider([_black()])
    pipeline = _make_pipeline(tmp_path, provider, quality_gate_max_retries=0)

    with pytest.raises(QualityGateError):
        pipeline.run(contents=["apple"], output_path=str(tmp_path / "out.pdf"))

    assert provider.call_count == 1
    mock_compile.assert_not_called()


# -- optional vision subject check --------------------------------------------


def _vision_analyzer(answer: str) -> MagicMock:
    analyzer = MagicMock()
    analyzer.analyze.return_value = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }
    analyzer.provider = MagicMock()
    analyzer.provider.generate_vision_content.return_value = answer
    return analyzer


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_vision_rejects_absent_subject(mock_compile, tmp_path: Path):
    """A well-formed illustration whose subject the vision model denies is
    rejected as "subject not present" and retried, then failed."""
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _ScriptedProvider([_gradient(512, 512) for _ in range(5)])
    pipeline = _make_pipeline(tmp_path, provider, vision_analyzer=_vision_analyzer("no"))

    with pytest.raises(QualityGateError) as exc:
        pipeline.run(contents=["apple"], output_path=str(tmp_path / "out.pdf"))

    assert "subject not present" in exc.value.reasons
    assert provider.call_count == 3
    mock_compile.assert_not_called()


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_vision_accepts_present_subject(mock_compile, tmp_path: Path):
    """A well-formed illustration whose subject the vision model confirms is
    accepted on the first attempt."""
    mock_compile.return_value = tmp_path / "out.pdf"
    provider = _ScriptedProvider([_gradient(512, 512)])
    pipeline = _make_pipeline(tmp_path, provider, vision_analyzer=_vision_analyzer("yes"))

    result = pipeline.run(contents=["apple"], output_path=str(tmp_path / "out.pdf"))

    assert isinstance(result, Path)
    assert provider.call_count == 1
    mock_compile.assert_called_once()
