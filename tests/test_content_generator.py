"""Tests for the optional LLM-based content generator."""

from unittest.mock import MagicMock, patch

import pytest

from pageloom.content_generator import LLMContentGenerator


def test_llm_content_generator_init_reads_env():
    with patch.dict(
        "os.environ", {"PAGELOOM_LLM_API_KEY": "test-key", "PAGELOOM_LLM_MODEL": "test-model"}
    ):
        gen = LLMContentGenerator()
        assert gen.api_key == "test-key"
        assert gen.model == "test-model"


def test_llm_content_generator_init_explicit_params():
    gen = LLMContentGenerator(api_key="explicit", model="mymodel")
    assert gen.api_key == "explicit"
    assert gen.model == "mymodel"


def test_llm_content_generator_generate_requires_openai():
    gen = LLMContentGenerator(api_key="test-key")
    import sys

    original_modules = dict(sys.modules)
    # Temporarily remove openai if it exists
    for k in list(sys.modules.keys()):
        if k == "openai" or k.startswith("openai."):
            del sys.modules[k]
    try:
        with pytest.raises(RuntimeError) as exc_info:
            gen.generate(3)
        assert "openai" in str(exc_info.value)
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_llm_content_generator_generate_uses_mock_openai():
    gen = LLMContentGenerator(api_key="test-key")
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Dragon, Unicorn, Whale"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        result = gen.generate(3, theme="animals")
        assert len(result) == 3
        assert result == ["Dragon", "Unicorn", "Whale"]
        mock_client.chat.completions.create.assert_called_once()
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_provider_switching_and_request_shapes():
    import sys

    # Test openai provider
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Apple, Banana"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        gen = LLMContentGenerator(provider="openai", api_key="test")
        result = gen.generate(2)
        assert result == ["Apple", "Banana"]
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert "messages" in call_args.kwargs
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)

    # Test openrouter provider (openai-compatible endpoint)
    mock_openai_module2 = MagicMock()
    mock_client2 = MagicMock()
    mock_response2 = MagicMock()
    mock_response2.choices = [MagicMock(message=MagicMock(content="Cherry, Dragon"))]
    mock_client2.chat.completions.create.return_value = mock_response2
    mock_openai_module2.OpenAI.return_value = mock_client2

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module2
    try:
        gen = LLMContentGenerator(provider="openrouter", api_key="test")
        result = gen.generate(2)
        assert result == ["Cherry", "Dragon"]
        call_args = mock_client2.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "openai/gpt-4o-mini"
        # OpenRouter uses its default endpoint
        assert gen.endpoint.startswith("https://openrouter.ai")
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)

    # Test anthropic provider
    mock_anthropic_module = MagicMock()
    mock_client_anthropic = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Fish, Grape"
    mock_response_anthropic = MagicMock()
    mock_response_anthropic.content = [mock_content]
    mock_client_anthropic.messages.create.return_value = mock_response_anthropic
    mock_anthropic_module.Anthropic.return_value = mock_client_anthropic

    original_modules = dict(sys.modules)
    sys.modules["anthropic"] = mock_anthropic_module
    try:
        gen = LLMContentGenerator(provider="anthropic", api_key="test")
        result = gen.generate(2)
        assert result == ["Fish", "Grape"]
        mock_client_anthropic.messages.create.assert_called_once()
        call_args = mock_client_anthropic.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-3-haiku-20240307"
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_provider_selection_via_env():
    with patch.dict(
        "os.environ",
        {"PAGELOOM_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"},
    ):
        gen = LLMContentGenerator()
        assert gen.provider_name == "anthropic"


def test_missing_openai_key_raises_clear_startup_error():
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="LLM provider 'openai' selected but API key is missing"),
    ):
        LLMContentGenerator(provider="openai")


def test_missing_anthropic_key_raises_clear_startup_error():
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(
            RuntimeError, match="LLM provider 'anthropic' selected but API key is missing"
        ),
    ):
        LLMContentGenerator(provider="anthropic")


def test_missing_openrouter_key_raises_clear_startup_error():
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(
            RuntimeError, match="LLM provider 'openrouter' selected but API key is missing"
        ),
    ):
        LLMContentGenerator(provider="openrouter")


def test_provider_credentials_read_from_env_vars():
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "openai-env",
            "ANTHROPIC_API_KEY": "anthropic-env",
            "OPENROUTER_API_KEY": "openrouter-env",
        },
    ):
        gen_openai = LLMContentGenerator(provider="openai")
        assert gen_openai.api_key == "openai-env"
        gen_anthropic = LLMContentGenerator(provider="anthropic")
        assert gen_anthropic.api_key == "anthropic-env"
        gen_openrouter = LLMContentGenerator(provider="openrouter")
        assert gen_openrouter.api_key == "openrouter-env"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMContentGenerator(provider="unknown")


def test_llm_content_generator_generate_returns_correct_length():
    gen = LLMContentGenerator(api_key="test-key")
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="A, B, C, D, E"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        result = gen.generate(2)
        assert len(result) == 2
        assert result[0] == "A"
        assert result[1] == "B"
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
