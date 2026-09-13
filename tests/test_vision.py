"""Tests for the vision-based page-structure analyzer (PageVisionAnalyzer)."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pageloom.vision import PageVisionAnalyzer, VisionAnalysisError


def _mock_openai_module(content: str) -> MagicMock:
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client
    return mock_openai_module, mock_client


def test_defaults_to_openrouter_provider():
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        analyzer = PageVisionAnalyzer()
        assert analyzer.provider_name == "openrouter"


def test_raises_clear_error_when_api_key_missing():
    with patch.dict(
        "os.environ",
        {},
        clear=True,
    ):
        with pytest.raises(VisionAnalysisError) as exc_info:
            PageVisionAnalyzer(provider="openrouter")
        assert "openrouter" in str(exc_info.value).lower()
        assert "OPENROUTER_API_KEY" in str(exc_info.value)


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError):
        PageVisionAnalyzer(provider="not-a-real-provider")


def test_analyze_parses_valid_json_response(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    valid_json = (
        '{"content_pattern": "number_count_matching", '
        '"content_type_description": "a page pairing a number with that many objects", '
        '"element_count": 3, "arrangement": "row", '
        '"text_image_relationship": "label_per_image", '
        '"example_content_items": ["3 ducks"]}'
    )
    mock_openai_module, mock_client = _mock_openai_module(valid_json)

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            result = analyzer.analyze(str(img_path))
        assert result["content_pattern"] == "number_count_matching"
        assert result["element_count"] == 3
        assert result["arrangement"] == "row"
        assert result["example_content_items"] == ["3 ducks"]
        # Confirm an actual multimodal request was sent: text + image_url content blocks.
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        content_blocks = call_kwargs["messages"][0]["content"]
        types = {block["type"] for block in content_blocks}
        assert types == {"text", "image_url"}
        assert content_blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_analyze_strips_markdown_code_fences(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    fenced_json = (
        "```json\n"
        '{"content_pattern": "digit_illustration", '
        '"content_type_description": "a page pairing a digit with an illustration", '
        '"element_count": 1, "arrangement": "centered", '
        '"text_image_relationship": "label_per_image", '
        '"example_content_items": ["7"]}'
        "\n```"
    )
    mock_openai_module, _ = _mock_openai_module(fenced_json)

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            result = analyzer.analyze(str(img_path))
        assert result["content_pattern"] == "digit_illustration"
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_analyze_raises_on_missing_required_field(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    incomplete_json = '{"content_pattern": "digit_illustration"}'
    mock_openai_module, _ = _mock_openai_module(incomplete_json)

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            with pytest.raises(VisionAnalysisError):
                analyzer.analyze(str(img_path))
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_analyze_raises_on_non_json_response(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    mock_openai_module, _ = _mock_openai_module("Sorry, I can't help with that.")

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            with pytest.raises(VisionAnalysisError):
                analyzer.analyze(str(img_path))
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_analyze_passes_description_hint_into_prompt(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    valid_json = (
        '{"content_pattern": "single_letter_word_illustration", '
        '"content_type_description": "a letter and word page", '
        '"element_count": 1, "arrangement": "centered", '
        '"text_image_relationship": "label_per_image", "example_content_items": ["Apple"]}'
    )
    mock_openai_module, mock_client = _mock_openai_module(valid_json)

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            analyzer.analyze(str(img_path), description="alphabet letter word")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        prompt_text = call_kwargs["messages"][0]["content"][0]["text"]
        assert "alphabet letter word" in prompt_text
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_smoke_real_image_and_live_openrouter_key():
    """Real smoke test: create a real PNG, initialize with live OPENROUTER_API_KEY,
    and confirm provider/model/endpoint are set correctly. The actual live call
    is verified separately (see smoke_vision.py); here we confirm the real image
    encodes correctly and the provider uses openrouter/free.
    """
    import os
    import tempfile

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    Image.new("RGB", (612, 612), color="red").save(img_path)

    try:
        key = os.environ.get("OPENROUTER_API_KEY")
        # This test requires a real key to initialize; skip if missing
        if not key:
            pytest.skip("No OPENROUTER_API_KEY set for smoke test")
        analyzer = PageVisionAnalyzer(provider="openrouter")
        assert analyzer.provider_name == "openrouter"
        # Confirm the free auto-router model is selected by default
        assert analyzer.provider.model == "openrouter/free"
        # Confirm image encoding works with a real file
        b64_url = analyzer._encode_image(img_path)
        assert b64_url.startswith("data:image/png;base64,")
    finally:
        os.unlink(img_path)


def test_vision_call_failure_wrapped_in_vision_analysis_error(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("network down")
    mock_openai_module.OpenAI.return_value = mock_client

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            with pytest.raises(VisionAnalysisError):
                analyzer.analyze(str(img_path))
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
