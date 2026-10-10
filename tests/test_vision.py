"""Tests for the vision-based page-structure analyzer (PageVisionAnalyzer)."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pageloom.vision import (
    _VISION_CACHE_DIR,
    PageVisionAnalyzer,
    VisionAnalysisError,
)


@pytest.fixture(autouse=True)
def clean_vision_cache():
    # Ensure each vision test starts with a clean cache directory so
    # mock-based tests don't accidentally read leftover cached results.
    import shutil

    if _VISION_CACHE_DIR.exists():
        shutil.rmtree(_VISION_CACHE_DIR)
    yield
    if _VISION_CACHE_DIR.exists():
        shutil.rmtree(_VISION_CACHE_DIR)


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


def test_vision_analysis_cache_reads_from_disk(tmp_path):
    """Cached vision results should be read without calling the API again."""
    from pageloom.vision import _VISION_CACHE_DIR

    # Pre-populate a fake cache file with the exact hash key.
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    # Import the analyzer class directly for key computation
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        analyzer = PageVisionAnalyzer(provider="openrouter")
        key = analyzer._cache_key(str(img_path), description="")
    cache_file = _VISION_CACHE_DIR / (key + ".json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fake_result = {
        "content_pattern": "cached_pattern",
        "content_type_description": "cached",
        "element_count": 42,
        "arrangement": "grid",
        "text_image_relationship": "label_per_image",
        "example_content_items": ["cached item"],
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        import json

        json.dump(fake_result, f)

    # Now initialize a fresh analyzer and call analyze -- it should return
    # the cached result without making any network call.
    mock_openai_module, mock_client = _mock_openai_module("should not be called")
    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            result = analyzer.analyze(str(img_path), description="")
        assert result == fake_result
        # No network call was made because the cache was hit.
        assert mock_client.chat.completions.create.call_count == 0
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
        # Clean up cached file
        if cache_file.exists():
            cache_file.unlink()


def test_vision_analysis_cache_writes_result(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="green").save(img_path)

    valid_json = (
        '{"content_pattern": "new_pattern", '
        '"content_type_description": "new desc", '
        '"element_count": 2, "arrangement": "row", '
        '"text_image_relationship": "label_per_image", '
        '"example_content_items": ["new"]}'
    )
    mock_openai_module, _ = _mock_openai_module(valid_json)

    from pageloom.vision import _VISION_CACHE_DIR

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            analyzer = PageVisionAnalyzer(provider="openrouter")
            result = analyzer.analyze(str(img_path), description="test desc")
        assert result["content_pattern"] == "new_pattern"
        key = analyzer._cache_key(str(img_path), description="test desc")
        cache_file = _VISION_CACHE_DIR / (key + ".json")
        assert cache_file.exists()
        import json

        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        assert cached["content_pattern"] == "new_pattern"
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
        # Clean up
        from pageloom.vision import _VISION_CACHE_DIR

        # We already computed `key` inside the try block; reuse it here.
        # If key isn't defined (e.g. exception before assignment), skip cleanup.
        try:
            cache_file_clean = _VISION_CACHE_DIR / (key + ".json")
            if cache_file_clean.exists():
                cache_file_clean.unlink()
        except NameError:
            pass


def test_vision_analysis_cache_different_description_different_key(tmp_path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="red").save(img_path)

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        analyzer = PageVisionAnalyzer(provider="openrouter")
        key1 = analyzer._cache_key(str(img_path), description="a")
        key2 = analyzer._cache_key(str(img_path), description="b")
    assert key1 != key2


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


class _FakeVisionProvider:
    """Stand-in for LLMProvider that replays scripted responses, so the
    analyze-level retry loop can be tested in isolation from the real
    provider's internal 429 backoff."""

    provider_name = "openrouter"
    model = "openrouter/free"
    endpoint = "https://openrouter.ai/api/v1"
    api_key = "test-key"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_vision_content(self, prompt, image_data_url, max_tokens, temperature=0.2):
        self.calls.append({"prompt": prompt, "image_data_url": image_data_url})
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _analyzer_with_fake_provider(responses):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        analyzer = PageVisionAnalyzer(provider="openrouter")
    analyzer.provider = _FakeVisionProvider(responses)
    return analyzer, analyzer.provider


_VALID_RETRY_JSON = (
    '{"content_pattern": "number_count_matching", '
    '"content_type_description": "a page pairing a number with objects", '
    '"element_count": 3, "arrangement": "row", '
    '"text_image_relationship": "label_per_image", '
    '"example_content_items": ["3 ducks"]}'
)


def test_analyze_retries_with_stricter_prompt_on_empty_content(tmp_path):
    """An 'empty content' response (openrouter/free may route to a non-vision
    model that returns nothing) triggers one retry with a stricter JSON-only
    prompt; if the retry returns valid JSON, analyze succeeds."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    analyzer, fake = _analyzer_with_fake_provider(
        [RuntimeError("Vision model returned empty content"), _VALID_RETRY_JSON]
    )
    result = analyzer.analyze(str(img_path))
    assert result["content_pattern"] == "number_count_matching"
    assert result["element_count"] == 3
    assert result["example_content_items"] == ["3 ducks"]
    assert len(fake.calls) == 2
    assert "MUST reply with ONLY a single JSON object" in fake.calls[1]["prompt"]
    assert "MUST reply with ONLY a single JSON object" not in fake.calls[0]["prompt"]


def test_analyze_retries_on_non_json_then_succeeds(tmp_path):
    """A non-JSON (prose) response triggers a retry that returns valid JSON."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    analyzer, fake = _analyzer_with_fake_provider(["Sorry, I can't help.", _VALID_RETRY_JSON])
    result = analyzer.analyze(str(img_path))
    assert result["content_pattern"] == "number_count_matching"
    assert result["element_count"] == 3
    assert len(fake.calls) == 2


def test_analyze_exhausts_retry_on_persistent_empty_content(tmp_path):
    """If both attempts return empty content, VisionAnalysisError is raised so the
    caller can fall back to the heuristic classifier."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    analyzer, fake = _analyzer_with_fake_provider(
        [
            RuntimeError("Vision model returned empty content"),
            RuntimeError("Vision model returned empty content"),
        ]
    )
    with pytest.raises(VisionAnalysisError) as exc_info:
        analyzer.analyze(str(img_path))
    assert "empty content" in str(exc_info.value).lower()
    assert len(fake.calls) == 2


def test_analyze_does_not_retry_on_rate_limit_error(tmp_path):
    """A rate-limit error is surfaced immediately -- a stricter prompt won't fix a
    rate limit, so analyze must not burn an analysis-level retry on it."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="blue").save(img_path)

    analyzer, fake = _analyzer_with_fake_provider(
        [RuntimeError("429 rate limit: free-models-per-day-high-balance")]
    )
    with pytest.raises(VisionAnalysisError) as exc_info:
        analyzer.analyze(str(img_path))
    assert len(fake.calls) == 1
    assert "rate limit" in str(exc_info.value).lower()
