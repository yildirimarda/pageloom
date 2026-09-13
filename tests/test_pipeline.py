from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pageloom.pipeline import Pipeline


def test_pipeline_init_with_example_image(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)
    mock_gen = MagicMock()
    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    assert pipeline.extractor is not None
    assert pipeline.style is not None


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
