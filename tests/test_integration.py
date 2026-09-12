"""End-to-end integration test: feeds one example page and generates a multi-page PDF."""

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from pypdf import PdfReader

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
