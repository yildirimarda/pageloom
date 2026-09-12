from pathlib import Path

from PIL import Image

from pageloom.style import StylePromptBuilder


def test_style_prompt_builder_template():
    builder = StylePromptBuilder("whimsical watercolor, soft pastel palette, gentle curves")
    template = builder.template()
    assert "whimsical watercolor" in template
    assert "{content}" in template


def test_style_prompt_builder_build():
    builder = StylePromptBuilder("whimsical watercolor, soft pastel palette, gentle curves")
    prompt = builder.build(content="apple", theme="nature")
    assert "whimsical watercolor" in prompt
    assert "content: apple" in prompt
    assert "theme: nature" in prompt
    assert prompt.endswith(".")


def test_style_prompt_builder_empty_description():
    builder = StylePromptBuilder("")
    prompt = builder.build(content="ball")
    assert "content: ball" in prompt
    assert "style reference" not in prompt


def test_style_prompt_builder_from_example_image(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (400, 400), color="blue")
    img.save(img_path)
    builder = StylePromptBuilder(example_image=str(img_path))
    assert builder.example_style is not None
    prompt = builder.build(content="cat")
    assert "content: cat" in prompt
    assert "inferred layout" in prompt or "centered_text_with_illustration" in prompt
