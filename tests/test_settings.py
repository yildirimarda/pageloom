"""Tests for environment-variable overrides in pageloom settings."""

from unittest.mock import MagicMock, patch

from pageloom import settings


def test_get_model_id_default():
    assert settings.get_model_id() == "runwayml/stable-diffusion-v1-5"


def test_get_model_id_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_MODEL_ID", "custom/model")
    assert settings.get_model_id() == "custom/model"


def test_get_device_default():
    assert settings.get_device() is None


def test_get_device_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_DEVICE", "cpu")
    assert settings.get_device() == "cpu"


def test_get_output_dir_default():
    assert settings.get_output_dir() is None


def test_get_output_dir_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_OUTPUT_DIR", "/tmp/output")
    assert settings.get_output_dir() == "/tmp/output"


def test_get_hf_cache_dir_default():
    assert settings.get_hf_cache_dir() is None


def test_get_hf_cache_dir_from_page_loom_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_HF_CACHE_DIR", "/tmp/hf_cache")
    assert settings.get_hf_cache_dir() == "/tmp/hf_cache"


def test_get_hf_cache_dir_from_hf_home_env(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/tmp/hf_home")
    assert settings.get_hf_cache_dir() == "/tmp/hf_home"


def test_get_hf_cache_dir_page_loom_takes_precedence(monkeypatch):
    monkeypatch.setenv("PAGELOOM_HF_CACHE_DIR", "/tmp/page_cache")
    monkeypatch.setenv("HF_HOME", "/tmp/hf_home")
    assert settings.get_hf_cache_dir() == "/tmp/page_cache"


def test_get_openai_llm_model_default():
    assert settings.get_openai_llm_model() == "gpt-4o-mini"


def test_get_openai_llm_model_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_LLM_MODEL", "custom-openai-model")
    assert settings.get_openai_llm_model() == "custom-openai-model"


def test_get_anthropic_llm_model_default():
    assert settings.get_anthropic_llm_model() == "claude-3-haiku-20240307"


def test_get_anthropic_llm_model_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_LLM_MODEL", "custom-claude-model")
    assert settings.get_anthropic_llm_model() == "custom-claude-model"


def test_get_openrouter_llm_model_default():
    assert settings.get_openrouter_llm_model() == "openai/gpt-4o-mini"


def test_get_openrouter_llm_model_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "custom-openrouter-model")
    assert settings.get_openrouter_llm_model() == "custom-openrouter-model"


def test_illustration_generator_uses_env_model_id(monkeypatch):
    monkeypatch.setenv("PAGELOOM_MODEL_ID", "env/model")
    with (
        patch("pageloom.generator.StableDiffusionPipeline.from_pretrained") as mock_from,
        patch("pageloom.generator.torch.cuda.is_available", return_value=False),
        patch("pageloom.generator.torch.backends.mps.is_available", return_value=False),
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = lambda device: mock_pipeline
        mock_from.return_value = mock_pipeline
        from pageloom.generator import IllustrationGenerator

        gen = IllustrationGenerator(verify_license=False)
        assert gen.model_id == "env/model"
        mock_from.assert_called_once()
        call_kwargs = mock_from.call_args.kwargs
        assert call_kwargs.get("cache_dir") is None or "env/model" in str(mock_from.call_args)


def test_illustration_generator_uses_env_device(monkeypatch):
    monkeypatch.setenv("PAGELOOM_DEVICE", "mps")
    with (
        patch("pageloom.generator.StableDiffusionPipeline.from_pretrained") as mock_from,
        patch("pageloom.generator.torch.cuda.is_available", return_value=False),
        patch("pageloom.generator.torch.backends.mps.is_available", return_value=True),
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = lambda device: mock_pipeline
        mock_from.return_value = mock_pipeline
        from pageloom.generator import IllustrationGenerator

        gen = IllustrationGenerator(verify_license=False)
        assert gen.device == "mps"
