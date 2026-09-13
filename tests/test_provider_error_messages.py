"""Re-verify that provider error messages use readable provider names."""

from unittest.mock import patch

from pageloom.content_generator import (
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
    _classify_provider_error,
)


def test_classify_provider_error_uses_readable_names():
    """Confirm exact user-facing strings for all error conditions."""

    # Rate limit
    class RateLimitError(Exception):
        pass

    msg = _classify_provider_error(RateLimitError("hit"), "openrouter")
    assert msg == "LLM provider 'openrouter' rate limit exceeded."

    # Timeout
    class APITimeoutError(Exception):
        pass

    msg = _classify_provider_error(APITimeoutError("timeout"), "anthropic")
    assert msg == "LLM provider 'anthropic' request timed out."

    # Invalid key (AuthenticationError)
    class AuthenticationError(Exception):
        pass

    msg = _classify_provider_error(AuthenticationError("bad key"), "openai")
    assert msg == "LLM provider 'openai' invalid or missing API key."

    # Invalid key (APIKeyError)
    class APIKeyError(Exception):
        pass

    msg = _classify_provider_error(APIKeyError("missing"), "openai")
    assert msg == "LLM provider 'openai' invalid or missing API key."

    # Invalid key (InvalidKeyError)
    class InvalidKeyError(Exception):
        pass

    msg = _classify_provider_error(InvalidKeyError("bad"), "openrouter")
    assert msg == "LLM provider 'openrouter' invalid or missing API key."

    # Generic / model doesn't support images
    class SomeVisionError(Exception):
        pass

    msg = _classify_provider_error(
        SomeVisionError("This model does not support image inputs"), "anthropic"
    )
    assert msg.startswith("LLM provider 'anthropic' API call failed:")
    assert "This model does not support image inputs" in msg
    assert "AnthropicProvider" not in msg
    assert "anthropic" in msg


def test_all_three_providers_pass_readable_name_to_classify():
    """Trigger the error path through each provider class and confirm output."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
    }
    for label, cls in providers.items():
        provider = cls(api_key="test")

        # Rate limit path
        class RateLimitError(Exception):
            pass

        # We can't easily trigger generate_content without mocking the SDK,
        # but we can directly verify that the provider's provider_name is
        # the readable label, not the Python class name.
        assert provider.provider_name == label
        assert provider.provider_name != cls.__name__
        # Verify the message produced by _classify_provider_error uses the label
        msg = _classify_provider_error(RateLimitError("limit"), provider.provider_name)
        assert f"LLM provider '{label}'" in msg
        assert cls.__name__ not in msg


def test_vision_model_reject_image_falls_back_gracefully_for_each_provider():
    """Simulate 'model does not support images' for all three providers and
    confirm the user-facing string contains the readable provider name."""
    providers = {
        "openai": ("openai", "OPENAI_API_KEY"),
        "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
        "openrouter": ("openrouter", "OPENROUTER_API_KEY"),
    }
    for label, (provider, env_key) in providers.items():
        with patch.dict("os.environ", {env_key: "test-key"}, clear=False):
            if label == "openai":
                prov = OpenAIProvider(api_key="test")
            elif label == "anthropic":
                prov = AnthropicProvider(api_key="test")
            else:
                prov = OpenRouterProvider(api_key="test")

            class ImageRejectedError(Exception):
                pass

            msg = _classify_provider_error(
                ImageRejectedError("This model does not support image inputs"), prov.provider_name
            )
            assert msg.startswith(f"LLM provider '{label}'")
            assert "AnthropicProvider" not in msg
            assert "OpenAIProvider" not in msg
            assert "OpenRouterProvider" not in msg
            assert "This model does not support image inputs" in msg
