"""Test that 429 rate limits are handled gracefully with retry/backoff."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from pageloom.content_generator import OpenRouterProvider, _retry_with_backoff


def test_retry_with_backoff_retries_on_rate_limit_then_succeeds():
    """If first call hits 429 and second succeeds, retry returns result."""
    call_count = 0

    def _fn():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("429 Too Many Requests")
        return "success"

    # Patch time.sleep to avoid real delays in tests.
    with patch("pageloom.content_generator.time.sleep"):
        result = _retry_with_backoff(_fn, max_retries=2, rate_limit_delay=0.1)
    assert result == "success"
    assert call_count == 2


def test_retry_with_backoff_raises_after_exhausting_retries():
    """If every attempt hits 429, the final exception propagates gracefully."""
    with (
        patch("pageloom.content_generator.time.sleep"),
        pytest.raises(RuntimeError, match="429"),
    ):
        def _fn():
            raise RuntimeError("429 free-models-per-day-high-balance")

        _retry_with_backoff(_fn, max_retries=2, rate_limit_delay=0.1)


def test_retry_does_not_retry_non_rate_limit_errors():
    """Non-rate-limit errors (e.g. bad key) are not retried."""
    call_count = 0

    def _fn():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("invalid key")

    with (
        patch("pageloom.content_generator.time.sleep"),
        pytest.raises(RuntimeError, match="invalid key"),
    ):
        _retry_with_backoff(_fn, max_retries=2, rate_limit_delay=0.1)
    assert call_count == 1


def test_openrouter_provider_vision_content_retries_on_429():
    """OpenRouter vision call retries gracefully on a 429 error."""
    mock_openai_module = MagicMock()
    mock_client = MagicMock()

    # First call raises 429, second succeeds.
    mock_client.chat.completions.create.side_effect = [
        RuntimeError("429 rate limit: free-models-per-day-high-balance"),
        MagicMock(choices=[MagicMock(message=MagicMock(content='{"content_pattern":"test"}'))]),
    ]
    mock_openai_module.OpenAI.return_value = mock_client

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterProvider(api_key="test-key", model="openrouter/free")
            with patch("pageloom.content_generator.time.sleep"):
                result = provider.generate_vision_content(
                    prompt="test",
                    image_data_url="data:image/png;base64,abc",
                    max_tokens=100,
                )
        assert '"content_pattern":"test"' in result
        assert mock_client.chat.completions.create.call_count == 2
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
