from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pageloom.consistency import ConsistencyGuard
from pageloom.pipeline import Pipeline, round_to_multiple
from pageloom.providers import ImageProvider, ImageProviderCapabilities


def test_round_to_multiple_returns_multiple_of_8():
    assert round_to_multiple(618) == 624  # the exact crash case from the bug report
    assert round_to_multiple(800) == 800
    assert round_to_multiple(612) == 616  # existing default page size was not divisible by 8
    assert round_to_multiple(624) == 624  # already a multiple
    assert round_to_multiple(0) == 0
    assert round_to_multiple(1) == 8
    for dim in range(100):
        assert round_to_multiple(dim) % 8 == 0
        assert 0 <= round_to_multiple(dim) - dim < 8


def test_round_to_multiple_rejects_non_positive_base():
    try:
        round_to_multiple(100, base=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for base=0")


def test_round_to_multiple_feeds_real_diffusers_check_inputs():
    """The helper's output must satisfy the *actual* diffusers VAE guard, not
    just our own % 8 assertion -- if diffusers ever tightens the rule, this
    catches it (Milestone 8 regression)."""
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
        StableDiffusionPipeline,
    )

    # Exact crash inputs from the bug report: 618x800.
    w, h = round_to_multiple(618), round_to_multiple(800)
    assert (w, h) == (624, 800)
    StableDiffusionPipeline.check_inputs(None, "prompt", h, w, 1)

    # The un-rounded inputs must be what diffusers actually rejects, so the
    # helper is load-bearing rather than decorative.
    with pytest.raises(ValueError, match="divisible by 8"):
        StableDiffusionPipeline.check_inputs(None, "prompt", 800, 618, 1)

    for dim in (618, 612, 151, 266, 1):
        StableDiffusionPipeline.check_inputs(
            None, "prompt", round_to_multiple(dim), round_to_multiple(dim), 1
        )


def test_pipeline_detects_page_structure_from_example(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (612, 612), color="blue")
    img.save(img_path)
    mock_gen = MagicMock()
    pipeline = Pipeline(
        example_image=str(img_path), description="alphabet letter word", generator=mock_gen
    )
    assert pipeline.page_structure is not None
    assert pipeline.page_structure.get("content_pattern") == "single_letter_word_illustration"


def test_pipeline_uses_description_for_object_structure(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (800, 400), color="green")
    img.save(img_path)
    mock_gen = MagicMock()
    pipeline = Pipeline(
        example_image=str(img_path), description="object name pair", generator=mock_gen
    )
    assert pipeline.page_structure.get("content_pattern") == "object_name_pair"


def test_pipeline_derives_format_from_structure(tmp_path: Path):
    img_path = tmp_path / "object_example.png"
    img = Image.new("RGB", (800, 400), color="green")
    img.save(img_path)
    mock_gen = MagicMock()
    pipeline = Pipeline(
        example_image=str(img_path), description="object name pair", generator=mock_gen
    )
    # When standard is not overridden, the pipeline should derive object_name
    assert pipeline.page_structure.get("content_pattern") == "object_name_pair"


def test_pipeline_init_with_example_image(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)
    mock_gen = MagicMock()
    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    assert pipeline.extractor is not None
    assert pipeline.style is not None


def test_pipeline_passes_text_image_relationship_to_overlay(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="green")
    img.save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (512, 512), color="red")

    # Force page_structure with no_text so overlay receives it
    pipeline = Pipeline(
        example_image=str(img_path),
        generator=mock_gen,
        page_structure={
            "content_pattern": "multi_item_scene",
            "element_count": 3,
            "arrangement": "grid",
            "text_image_relationship": "no_text",
        },
    )
    with patch("pageloom.pipeline.PDFCompiler.compile") as mock_compile:
        mock_compile.return_value = tmp_path / "output.pdf"
        pipeline.run(
            contents=["apple"],
            output_path=str(tmp_path / "output.pdf"),
        )
        # The overlay should have been called; if relationship is no_text,
        # overlay.apply returns image without drawing text.
        assert mock_compile.called


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_run(mock_compile, tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="green")
    img.save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (512, 512), color="red")
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    result = pipeline.run(
        contents=["apple", "banana"],
        themes=["nature", ""],
        output_path=str(tmp_path / "output.pdf"),
    )
    assert isinstance(result, Path)
    assert mock_gen.generate.call_count == 2
    assert mock_compile.called


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_run_caches_raw_illustrations(mock_compile, tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="green")
    img.save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (512, 512), color="red")
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    result = pipeline.run(
        contents=["apple"],
        output_path=str(tmp_path / "output.pdf"),
    )
    assert isinstance(result, Path)
    assert mock_gen.generate.call_count == 1
    # Rerun without --force should skip generation
    result2 = pipeline.run(
        contents=["apple"],
        output_path=str(tmp_path / "output.pdf"),
    )
    assert isinstance(result2, Path)
    assert mock_gen.generate.call_count == 1  # not called again


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_run_force_regenerates(mock_compile, tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="green")
    img.save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (512, 512), color="blue")
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    pipeline.run(
        contents=["apple"],
        output_path=str(tmp_path / "output.pdf"),
    )
    assert mock_gen.generate.call_count == 1

    # Force should regenerate even though cached
    result = pipeline.run(
        contents=["apple"],
        output_path=str(tmp_path / "output.pdf"),
        force=True,
    )
    assert isinstance(result, Path)
    assert mock_gen.generate.call_count == 2


def make_sd_checked_image(color: str = "red"):
    """Mock-generator side effect that runs diffusers' *real* dimension check.

    A plain MagicMock ignores whatever width/height it is called with, which is
    exactly how the Milestone 8 618x800 crash stayed invisible to every test.
    Running ``check_inputs`` here fails the test (with diffusers' own message)
    if the pipeline ever stops rounding, without loading the model.
    """

    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
        StableDiffusionPipeline,
    )

    def _generate(prompt: str, **kwargs) -> Image.Image:
        w = int(kwargs.get("width", 512))
        h = int(kwargs.get("height", 512))
        StableDiffusionPipeline.check_inputs(None, prompt, h, w, 1)
        return Image.new("RGB", (w, h), color=color)

    return _generate


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_generate_dims_divisible_by_8_for_odd_source_size(mock_compile, tmp_path: Path):
    """A 618x800 example (618 not divisible by 8) must never reach the
    generator with non-multiple-of-8 dimensions, or real Stable Diffusion's VAE
    check_inputs() would raise ValueError (Milestone 8 regression).

    The mock generator runs diffusers' actual check_inputs, so this fails if
    the rounding is removed -- unlike a MagicMock, it cannot swallow the bug.
    """
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (618, 800), color="green")
    img.save(img_path)

    called_with: list[tuple[int, int]] = []

    def make_image(prompt: str, **kwargs) -> Image.Image:
        w = int(kwargs.get("width", 512))
        h = int(kwargs.get("height", 512))
        called_with.append((w, h))
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
            StableDiffusionPipeline,
        )

        StableDiffusionPipeline.check_inputs(None, prompt, h, w, 1)
        return Image.new("RGB", (w, h), color="red")

    mock_gen = MagicMock()
    mock_gen.generate.side_effect = make_image
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    pipeline.run(
        contents=["apple", "banana"],
        output_path=str(tmp_path / "output.pdf"),
    )

    assert mock_gen.generate.call_count == 2
    assert called_with, "generator should have been called"
    # 618 -> 624 (round up), 800 stays as-is; both must pass the real VAE check.
    assert called_with == [(624, 800), (624, 800)], (
        f"expected rounded 624x800 page size, got {called_with}"
    )
    for w, h in called_with:
        assert w % 8 == 0 and h % 8 == 0, f"generator called with non-multiple-of-8 size {w}x{h}"


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_multiple_labels_sub_images_dims_divisible_by_8(mock_compile, tmp_path: Path):
    """The multi-sub-image path derives sub-image sizes via division, which can
    produce non-multiples of 8 (e.g. 800 // 3 == 266); those must also be
    rounded before reaching the generator."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (618, 800), color="yellow")
    img.save(img_path)

    mock_gen = MagicMock()
    mock_gen.generate.side_effect = make_sd_checked_image("blue")
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(
        example_image=str(img_path),
        generator=mock_gen,
        page_structure={
            "content_pattern": "multi_item_scene",
            "element_count": 3,
            "arrangement": "column",
            "text_image_relationship": "multiple_labels",
        },
    )
    pipeline.run(
        contents=["apple, banana, cherry"],
        output_path=str(tmp_path / "output.pdf"),
    )

    assert mock_gen.generate.call_count == 3  # one generate per element


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_run_verifies_generated_pages(mock_compile, tmp_path: Path, capsys):
    """run() must construct a real ConsistencyGuard and verify every generated
    page against the example, recording one result per page and reporting a
    summary -- previously ConsistencyGuard was only ever built inside tests,
    so no real run was consistency-checked."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (512, 512), color="red")
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    assert isinstance(pipeline.consistency_guard, ConsistencyGuard)

    result = pipeline.run(
        contents=["apple", "banana"],
        output_path=str(tmp_path / "output.pdf"),
    )

    assert result == tmp_path / "output.pdf"
    assert len(pipeline.consistency_results) == 2
    for page_result in pipeline.consistency_results:
        assert "consistent" in page_result
        assert "issues" in page_result
        assert "details" in page_result
    out = capsys.readouterr()
    assert "Consistency check: " in out.out
    assert " page(s) match the example." in out.out


def test_pipeline_passes_vision_analyzer_to_consistency_guard(tmp_path: Path):
    """The pipeline's vision_analyzer must reach the guard so generated pages
    are re-classified with the same method used for the example."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    fake_vision = MagicMock()
    fake_vision.analyze.return_value = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }
    mock_gen = MagicMock()

    pipeline = Pipeline(
        example_image=str(img_path), generator=mock_gen, vision_analyzer=fake_vision
    )

    # The example structure was classified with the vision analyzer...
    fake_vision.analyze.assert_called_once()
    # ...and the guard keeps it to re-classify generated pages the same way.
    assert pipeline.consistency_guard.vision_analyzer is fake_vision


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_run_survives_consistency_guard_failure(mock_compile, tmp_path: Path):
    """An unexpected guard error must not abort generation; it is recorded as
    an inconsistency result instead (verification is a quality signal, not a
    hard requirement for producing the PDF)."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (512, 512), color="red")
    mock_compile.return_value = tmp_path / "output.pdf"

    with patch("pageloom.pipeline.ConsistencyGuard") as mock_guard_cls:
        mock_guard_cls.return_value.verify.side_effect = RuntimeError("boom")
        pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
        result = pipeline.run(
            contents=["apple"],
            output_path=str(tmp_path / "output.pdf"),
        )

    assert result == tmp_path / "output.pdf"
    assert len(pipeline.consistency_results) == 1
    failed = pipeline.consistency_results[0]
    assert failed["consistent"] is False
    assert "Consistency check failed" in failed["issues"][0]
    assert "boom" in failed["issues"][0]


def test_pipeline_verify_consistency_default_is_true(tmp_path: Path):
    """The opt-out must be opt-in: default runs still verify pages."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    pipeline = Pipeline(example_image=str(img_path), generator=MagicMock())
    assert pipeline.verify_consistency is True


def test_pipeline_depends_only_on_image_provider_interface(tmp_path: Path):
    """Pipeline.run() must work with any ImageProvider, not just the SD backend."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    class StubProvider:
        capabilities = ImageProviderCapabilities(
            max_reference_images=1, supports_negative_prompt=True
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
            self.calls.append({"prompt": prompt, "width": width, "height": height})
            return Image.new("RGB", (width, height), color="green")

        def estimated_cost_usd(self, width: int, height: int) -> float:
            return 0.0

    assert isinstance(StubProvider(), ImageProvider)
    provider = StubProvider()
    pipeline = Pipeline(example_image=str(img_path), generator=provider)
    assert pipeline.generator is provider


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_negotiates_provider_size_then_fits_to_slot(mock_compile, tmp_path: Path):
    """A provider that only supports 512x512 and 1024x1024 must generate at the
    nearest supported size, and the pipeline must crop/pad the result back to
    the layout slot (size negotiation, Milestone 9)."""
    mock_compile.return_value = tmp_path / "out.pdf"
    calls: list[tuple[int, int]] = []

    class CappedProvider:
        capabilities = ImageProviderCapabilities(supported_sizes=((512, 512), (1024, 1024)))

        def generate(
            self,
            prompt: str,
            width: int,
            height: int,
            references: list[str | Image.Image] | None = None,
            negative_prompt: str | None = None,
            seed: int | None = None,
        ) -> Image.Image:
            calls.append((width, height))
            return Image.new("RGB", (width, height), color="red")

        def estimated_cost_usd(self, width: int, height: int) -> float:
            return 0.0

    # 612 is not a multiple of 8: the page slot rounds up to 616x792.
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 792), color="yellow").save(img_path)
    provider = CappedProvider()
    pipeline = Pipeline(example_image=str(img_path), generator=provider)
    pipeline.run(
        contents=["apple"],
        output_path=str(tmp_path / "out.pdf"),
    )
    # Generation happened at the nearest supported size (512x512 by area),
    # not at the requested 616x792 slot size.
    assert calls == [(512, 512)], f"expected generation at 512x512, got {calls}"
    # The compiled page is exactly the layout slot size.
    pages = mock_compile.call_args.args[0]
    assert pages[0].size == (616, 792)


class _ReferenceSupportingProvider:
    """Fake provider that accepts reference images and records every call."""

    capabilities = ImageProviderCapabilities(max_reference_images=3)

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
            }
        )
        return Image.new("RGB", (width, height), color="red")

    def estimated_cost_usd(self, width: int, height: int) -> float:
        return 0.0


class _NoReferenceProvider:
    """Fake provider without reference support: must keep the text path."""

    capabilities = ImageProviderCapabilities(max_reference_images=0)

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
            }
        )
        return Image.new("RGB", (width, height), color="green")

    def estimated_cost_usd(self, width: int, height: int) -> float:
        return 0.0


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_sends_references_when_provider_supports_them(mock_compile, tmp_path: Path):
    """Provider with max_reference_images > 0: page 1 is text-only, later pages
    reference the example page and the first generated page with a
    match-style instruction instead of the text-only style description."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    mock_compile.return_value = tmp_path / "output.pdf"
    provider = _ReferenceSupportingProvider()

    pipeline = Pipeline(example_image=str(img_path), generator=provider)
    pipeline.run(
        contents=["apple", "banana", "cherry"],
        output_path=str(tmp_path / "output.pdf"),
    )

    assert len(provider.calls) == 3
    first_call = provider.calls[0]
    # Page 1: no references exist yet -> text-only path.
    assert first_call["references"] is None
    assert "style reference:" in first_call["prompt"]
    assert "match the style of the reference images" not in first_call["prompt"]

    # The first generated page is remembered for use as a reference.
    assert pipeline._first_generated_page is not None
    assert pipeline._first_generated_page.size == (512, 512)

    # Pages 2+: references = [example page, first generated page],
    # prompt instructs to match the reference style.
    for call in provider.calls[1:]:
        refs = call["references"]
        assert refs is not None
        assert len(refs) == 2
        assert refs[0] is pipeline.extractor.image
        assert refs[1] is pipeline._first_generated_page
        assert "match the style of the reference images" in call["prompt"]
        assert "style reference:" not in call["prompt"]


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_keeps_text_path_when_provider_lacks_reference_support(
    mock_compile, tmp_path: Path
):
    """Provider with max_reference_images == 0: all pages stay on the text-only
    path, no references are ever sent."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    mock_compile.return_value = tmp_path / "output.pdf"
    provider = _NoReferenceProvider()

    pipeline = Pipeline(example_image=str(img_path), generator=provider)
    pipeline.run(
        contents=["apple", "banana"],
        output_path=str(tmp_path / "output.pdf"),
    )

    assert len(provider.calls) == 2
    for call in provider.calls:
        assert call["references"] is None
        assert "match the style of the reference images" not in call["prompt"]
        assert "style reference:" in call["prompt"]
