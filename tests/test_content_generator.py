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


def test_all_three_providers_called_with_correct_request_shape_and_switching():
    import sys

    # OpenAI provider request shape
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="A, B"))]
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai.OpenAI.return_value = mock_client

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai
    try:
        gen = LLMContentGenerator(provider="openai", api_key="test")
        result = gen.generate(2, theme="test-theme")
        assert result == ["A", "B"]
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100  # max(100, 2*20)
        assert isinstance(call_kwargs["messages"], list)
        assert call_kwargs["messages"][0]["role"] == "user"
        assert "test-theme" in call_kwargs["messages"][0]["content"]
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)

    # Anthropic provider request shape
    mock_anthropic = MagicMock()
    mock_client_a = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "C, D"
    mock_resp_a = MagicMock()
    mock_resp_a.content = [mock_content]
    mock_client_a.messages.create.return_value = mock_resp_a
    mock_anthropic.Anthropic.return_value = mock_client_a

    original_modules = dict(sys.modules)
    sys.modules["anthropic"] = mock_anthropic
    try:
        gen = LLMContentGenerator(provider="anthropic", api_key="test")
        result = gen.generate(2, theme="animals")
        assert result == ["C", "D"]
        call_kwargs = mock_client_a.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-haiku-20240307"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100  # max(100, 2*20)
        assert isinstance(call_kwargs["messages"], list)
        assert call_kwargs["messages"][0]["role"] == "user"
        assert "animals" in call_kwargs["messages"][0]["content"]
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)

    # OpenRouter provider request shape
    mock_openai2 = MagicMock()
    mock_client2 = MagicMock()
    mock_resp2 = MagicMock()
    mock_resp2.choices = [MagicMock(message=MagicMock(content="E, F"))]
    mock_client2.chat.completions.create.return_value = mock_resp2
    mock_openai2.OpenAI.return_value = mock_client2

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai2
    try:
        gen = LLMContentGenerator(provider="openrouter", api_key="test")
        result = gen.generate(2, theme="nature")
        assert result == ["E", "F"]
        call_kwargs = mock_client2.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100  # max(100, 2*20)
        assert isinstance(call_kwargs["messages"], list)
        assert call_kwargs["messages"][0]["role"] == "user"
        assert "nature" in call_kwargs["messages"][0]["content"]
        assert gen.endpoint.startswith("https://openrouter.ai")
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)

    # Provider switching via environment/config
    with patch.dict(
        "os.environ",
        {"PAGELOOM_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "env-key"},
    ):
        gen_env = LLMContentGenerator()
        assert gen_env.provider_name == "anthropic"
        assert gen_env.api_key == "env-key"


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


def test_llm_api_rate_limit_raises_clear_error():
    gen = LLMContentGenerator(api_key="test-key")
    mock_openai_module = MagicMock()
    mock_client = MagicMock()

    class RateLimitError(Exception):
        pass

    mock_client.chat.completions.create.side_effect = RateLimitError("limit")
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            gen.generate(2)
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_llm_api_timeout_raises_clear_error():
    gen = LLMContentGenerator(api_key="test-key")
    mock_openai_module = MagicMock()
    mock_client = MagicMock()

    class APITimeoutError(Exception):
        pass

    mock_client.chat.completions.create.side_effect = APITimeoutError("timeout")
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            gen.generate(2)
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_llm_api_invalid_key_raises_clear_error():
    gen = LLMContentGenerator(api_key="test-key")
    mock_openai_module = MagicMock()
    mock_client = MagicMock()

    class AuthenticationError(Exception):
        pass

    mock_client.chat.completions.create.side_effect = AuthenticationError("bad key")
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with pytest.raises(RuntimeError, match="invalid or missing API key"):
            gen.generate(2)
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_llm_insufficient_content_raises_clear_error():
    gen = LLMContentGenerator(api_key="test-key")
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="A"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with pytest.raises(RuntimeError, match="returned insufficient content"):
            gen.generate(3)
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


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
