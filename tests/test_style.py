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


def test_style_prompt_builder_template_empty_description():
    builder = StylePromptBuilder("")
    template = builder.template()
    assert "content: {content}" in template


def test_style_prompt_builder_build_object_name_format():
    builder = StylePromptBuilder("whimsical watercolor")
    prompt = builder.build(content="apple", page_format="object_name")
    assert "Photo-realistic children's book illustration" in prompt
    assert "object name: apple" in prompt


def test_style_prompt_builder_build_themed_scene_format():
    builder = StylePromptBuilder("soft pastel")
    prompt = builder.build(content="garden", theme="autumn", page_format="themed_scene")
    assert "Themed children's book scene illustration" in prompt
    assert "theme: autumn" in prompt
    assert "scene: garden" in prompt


def test_style_prompt_builder_init_both_image_and_style_raises():
    try:
        StylePromptBuilder(example_image="fake.png", example_style={"composition": {}})
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Provide either" in str(exc)


def test_style_prompt_builder_build_with_example_style_dict():
    builder = StylePromptBuilder(
        example_style={"composition": {"layout_type": "centered_text_with_illustration"}}
    )
    prompt = builder.build(content="ball")
    assert "style reference: centered_text_with_illustration" in prompt


def test_style_prompt_builder_build_with_structure():
    builder = StylePromptBuilder("soft pastel")
    prompt = builder.build(
        content="apple",
        element_count=3,
        arrangement="grid",
        text_image_relationship="label_per_image",
    )
    assert "3 illustrated element(s)" in prompt
    assert "arranged in a grid layout" in prompt
    assert "each image paired with its label" in prompt


def test_style_prompt_builder_from_example_image(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (400, 400), color="blue")
    img.save(img_path)
    builder = StylePromptBuilder(example_image=str(img_path))
    assert builder.example_style is not None
    prompt = builder.build(content="cat")
    assert "content: cat" in prompt
    assert "inferred layout" in prompt or "centered_text_with_illustration" in prompt
