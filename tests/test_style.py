from pageloom.style import StylePromptBuilder


def test_style_prompt_builder_template():
    builder = StylePromptBuilder("whimsical watercolor, soft pastel palette, gentle curves")
    template = builder.template()
    assert "whimsical watercolor" in template
    assert "{letter}" in template
    assert "{word}" in template


def test_style_prompt_builder_build():
    builder = StylePromptBuilder("whimsical watercolor, soft pastel palette, gentle curves")
    prompt = builder.build(letter="A", word="apple", theme="nature")
    assert "whimsical watercolor" in prompt
    assert "letter: A" in prompt
    assert "word: apple" in prompt
    assert "theme: nature" in prompt
    assert prompt.endswith(".")


def test_style_prompt_builder_empty_description():
    builder = StylePromptBuilder("")
    prompt = builder.build(letter="B", word="ball")
    assert "letter: B" in prompt
    assert "word: ball" in prompt
    assert "style reference" not in prompt
