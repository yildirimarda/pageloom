from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pageloom.overlay import TextOverlay
from pageloom.pipeline import Pipeline
from pageloom.style import StylePromptBuilder


def test_style_prompt_builder_object_name():
    builder = StylePromptBuilder("soft watercolor")
    prompt = builder.build(content="apple", page_format="object_name")
    assert "Photo-realistic" in prompt
    assert "object name: apple" in prompt


def test_style_prompt_builder_themed_scene():
    builder = StylePromptBuilder("vibrant fantasy")
    prompt = builder.build(content="forest", theme="nature", page_format="themed_scene")
    assert "Themed" in prompt
    assert "theme: nature" in prompt
    assert "scene: forest" in prompt


def test_text_overlay_object_name():
    base = Image.new("RGB", (400, 400), color="white")
    overlay = TextOverlay()
    result = overlay.apply(base, text="apple", page_format="object_name")
    assert isinstance(result, Image.Image)
    assert result.size == (400, 400)


def test_text_overlay_themed_scene():
    base = Image.new("RGB", (400, 400), color="blue")
    overlay = TextOverlay()
    result = overlay.apply(base, text="forest", page_format="themed_scene")
    assert isinstance(result, Image.Image)
    assert result.size == (400, 400)


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_run_object_name(mock_compile, tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (400, 400), color="green")
    img.save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (400, 400), color="red")
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    result = pipeline.run(
        contents=["apple"], themes=[""], output_path=tmp_path / "out.pdf", page_format="object_name"
    )
    assert isinstance(result, Path)
    assert mock_gen.generate.call_count == 1
    prompt_arg = mock_gen.generate.call_args[1].get("prompt", "")
    assert "Photo-realistic" in prompt_arg


@patch("pageloom.pipeline.PDFCompiler.compile")
def test_pipeline_run_themed_scene(mock_compile, tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (400, 400), color="blue")
    img.save(img_path)
    mock_gen = MagicMock()
    mock_gen.generate.return_value = Image.new("RGB", (400, 400), color="red")
    mock_compile.return_value = tmp_path / "output.pdf"

    pipeline = Pipeline(example_image=str(img_path), generator=mock_gen)
    result = pipeline.run(
        contents=["sunset"],
        themes=["nature"],
        output_path=tmp_path / "out.pdf",
        page_format="themed_scene",
    )
    assert isinstance(result, Path)
    assert mock_gen.generate.call_count == 1
