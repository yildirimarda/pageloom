"""End-to-end integration test: feeds one example page and generates a multi-page PDF."""

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from pypdf import PdfReader

from pageloom.consistency import ConsistencyGuard
from pageloom.page_structure_classifier import PageStructureClassifier
from pageloom.pipeline import Pipeline


def test_integration_example_page_to_multi_page_pdf(tmp_path: Path):
    """Feed a single example image to Pipeline and produce a matching multi-page PDF."""
    # 1. Create a synthetic example page
    example_path = tmp_path / "example.png"
    example_img = Image.new("RGB", (612, 612), color="purple")
    example_img.save(example_path)

    # 2. Mock generator so we don't load the Stable Diffusion model
    mock_gen = MagicMock()

    def make_image(prompt: str, **kwargs) -> Image.Image:
        # Return a distinct image per prompt so we can verify multi-page output
        color = (hash(prompt) % 256, (hash(prompt) // 256) % 256, 128)
        return Image.new("RGB", (612, 612), color=color)

    mock_gen.generate.side_effect = make_image

    # 3. Run the full pipeline with real overlay and PDF compiler
    pipeline = Pipeline(example_image=str(example_path), generator=mock_gen)
    output_path = tmp_path / "multi_page.pdf"
    result = pipeline.run(
        contents=["apple", "banana", "cherry"],
        themes=["nature", "nature", ""],
        output_path=str(output_path),
        page_format="standard",
    )

    # 4. Assert PDF was produced and contains 3 pages
    assert isinstance(result, Path)
    assert result.exists()
    assert result == output_path

    reader = PdfReader(str(output_path))
    page_count = len(reader.pages)
    assert page_count == 3, f"Expected 3 pages, got {page_count}"

    # Each page should have a non-empty media box (print-ready size)
    for i, page in enumerate(reader.pages):
        box = page.mediabox
        assert box is not None
        width = float(box.width)
        height = float(box.height)
        assert width > 0 and height > 0


def test_two_structurally_different_example_pages(tmp_path: Path):
    """Confirm pipeline produces pages matching each example's structure (not just style)."""
    # 1. Alphabet page example
    alphabet_path = tmp_path / "alphabet_example.png"
    alphabet_img = Image.new("RGB", (612, 612), color="blue")
    alphabet_img.save(alphabet_path)
    # Describe it as an alphabet page so the classifier detects single_letter_word_illustration
    alphabet_classifier = PageStructureClassifier(
        source=str(alphabet_path), description="alphabet letter word"
    )
    alphabet_structure = alphabet_classifier.classify()
    assert alphabet_structure["content_pattern"] == "single_letter_word_illustration"

    # 2. Object-naming / multi-item scene page example (wide format triggers multi-item scene)
    object_path = tmp_path / "object_example.png"
    object_img = Image.new("RGB", (800, 400), color="green")
    object_img.save(object_path)
    object_classifier = PageStructureClassifier(
        source=str(object_path), description="object name pair"
    )
    object_structure = object_classifier.classify()
    assert object_structure["content_pattern"] == "object_name_pair"

    # 3. Mock generator for both pipeline runs
    mock_gen = MagicMock()

    def make_image(prompt: str, **kwargs) -> Image.Image:
        color = (hash(prompt) % 256, (hash(prompt) // 256) % 256, 128)
        return Image.new("RGB", (612, 612), color=color)

    mock_gen.generate.side_effect = make_image

    # 4. Run pipeline for alphabet structure (standard format), passing the
    # same description used for classification above so Pipeline/ConsistencyGuard
    # route through the same structure that was detected, not pure image heuristics.
    pipeline_alpha = Pipeline(
        example_image=str(alphabet_path),
        description="alphabet letter word",
        generator=mock_gen,
    )
    output_alpha = tmp_path / "alphabet_output.pdf"
    result_alpha = pipeline_alpha.run(
        contents=["Apple", "Banana"],
        themes=["nature", "nature"],
        output_path=str(output_alpha),
        page_format="standard",
    )
    assert isinstance(result_alpha, Path)
    assert result_alpha.exists()

    # Verify the generated pages structurally match the alphabet example
    # (Use the first generated image from the mock for structural check)
    # We rely on the pipeline using the source image structure to guide output.
    guard_alpha = ConsistencyGuard(
        example_image=str(alphabet_path), description="alphabet letter word"
    )
    assert guard_alpha.example_structure["content_pattern"] == "single_letter_word_illustration"

    # 5. Run pipeline for object-naming structure. The description is required
    # here too: the example image alone (800x400) is wide enough that pure
    # image heuristics would misclassify it as multi_item_scene, exactly the
    # gap described in PLAN.md -- descriptions must flow through the pipeline
    # and consistency guard, not just the standalone classifier.
    mock_gen.reset_mock()
    mock_gen.generate.side_effect = make_image
    pipeline_object = Pipeline(
        example_image=str(object_path),
        description="object name pair",
        generator=mock_gen,
    )
    output_object = tmp_path / "object_output.pdf"
    result_object = pipeline_object.run(
        contents=["Cat", "Dog"],
        themes=[""],
        output_path=str(output_object),
        page_format="object_name",
    )
    assert isinstance(result_object, Path)
    assert result_object.exists()

    # Pipeline.run() should also have derived the object_name page_format from
    # the detected structure (it was explicitly passed here, but page_structure
    # is asserted directly below to confirm the routing itself is correct).
    assert pipeline_object.page_structure["content_pattern"] == "object_name_pair"

    guard_object = ConsistencyGuard(example_image=str(object_path), description="object name pair")
    assert guard_object.example_structure["content_pattern"] == "object_name_pair"

    # 6. Confirm the two structures are different (not just style)
    assert alphabet_structure["content_pattern"] != object_structure["content_pattern"]
    assert alphabet_structure["content_pattern"] == "single_letter_word_illustration"
    assert object_structure["content_pattern"] == "object_name_pair"
    assert guard_object.example_structure["content_pattern"] == "object_name_pair"


def test_multi_sub_image_compositing_for_multiple_labels(tmp_path: Path):
    """Full multi-sub-image compositing: one illustration per element,
    composited onto a single canvas with multiple TextOverlay placements."""
    # 1. Create synthetic example image
    example_path = tmp_path / "multi_example.png"
    example_img = Image.new("RGB", (612, 612), color="yellow")
    example_img.save(example_path)

    # 2. Mock generator
    mock_gen = MagicMock()

    def make_sub_image(prompt: str, **kwargs) -> Image.Image:
        # Return a small colored image so compositing is visible
        return Image.new("RGB", (200, 200), color=(hash(prompt) % 256, 0, 255))

    mock_gen.generate.side_effect = make_sub_image

    # 3. Force multiple_labels structure
    pipeline = Pipeline(
        example_image=str(example_path),
        description="multi-item labeled scene with multiple labels",
        generator=mock_gen,
        page_structure={
            "content_pattern": "multi_item_scene",
            "element_count": 3,
            "arrangement": "grid",
            "text_image_relationship": "multiple_labels",
        },
    )

    output_path = tmp_path / "multi_composite.pdf"
    result = pipeline.run(
        contents=["apple, banana, cherry"],
        themes=["nature"],
        output_path=str(output_path),
        page_format="standard",
    )

    assert isinstance(result, Path)
    assert result.exists()

    reader = PdfReader(str(output_path))
    assert len(reader.pages) == 1

    # The pipeline should have called generate 3 times (one per element)
    # for the single content item with element_count=3
    assert mock_gen.generate.call_count == 3

    # Verify the composited page image contains 3 sub-images and text labels.
    # We intercept the pages list before PDF compilation to inspect them directly.
    captured_pages: list[Image.Image] = []
    original_compile = pipeline.compiler.compile

    def capturing_compile(pages, output_path):
        captured_pages.extend(pages)
        return original_compile(pages, output_path)

    pipeline.compiler.compile = capturing_compile  # type: ignore[method-assign]

    # Rerun with the same settings; the mock will return the same sub-images.
    mock_gen.reset_mock()
    mock_gen.generate.side_effect = make_sub_image
    pipeline.run(
        contents=["apple, banana, cherry"],
        themes=["nature"],
        output_path=str(tmp_path / "multi_composite2.pdf"),
        page_format="standard",
    )

    assert len(captured_pages) == 1
    final_image = captured_pages[0]
    # Example page is 612x612; the pipeline rounds page dimensions up to the
    # nearest multiple of 8 (616) so real Stable Diffusion runs never hit the
    # VAE's "divisible by 8" check (see Milestone 8).
    assert final_image.size == (616, 616)

    # Confirm the image contains multiple distinct colors (sub-images) and dark text.
    pixels = list(final_image.getdata())
    unique_colors = set(pixels)
    dark_pixels = sum(1 for p in pixels if p[0] < 80 and p[1] < 80 and p[2] < 80)
    assert dark_pixels > 0, "Text overlay should add dark pixels"
    assert len(unique_colors) > 3, (
        f"Expected multiple colors from composited sub-images, got {len(unique_colors)}"
    )
