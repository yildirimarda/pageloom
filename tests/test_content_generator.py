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
    gen = LLMContentGenerator()
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


def test_llm_content_generator_generate_returns_correct_length():
    gen = LLMContentGenerator()
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
