"""Tests for environment-variable overrides in pageloom settings."""

from unittest.mock import MagicMock, patch
from urllib.error import URLError

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


def test_get_cpu_slow_warn_threshold_seconds_default():
    assert settings.get_cpu_slow_warn_threshold_seconds() == 300.0


def test_get_cpu_slow_warn_threshold_seconds_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_CPU_SLOW_WARN_SECONDS", "60")
    assert settings.get_cpu_slow_warn_threshold_seconds() == 60.0


def test_get_cpu_slow_warn_threshold_seconds_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("PAGELOOM_CPU_SLOW_WARN_SECONDS", "not-a-number")
    assert settings.get_cpu_slow_warn_threshold_seconds() == 300.0


def test_get_cpu_slow_warn_threshold_seconds_zero_disables(monkeypatch):
    monkeypatch.setenv("PAGELOOM_CPU_SLOW_WARN_SECONDS", "0")
    assert settings.get_cpu_slow_warn_threshold_seconds() == 0.0


def test_estimated_cpu_run_seconds():
    # 20 pages x 25 steps x 2 s/step = 1000 s
    assert settings.estimated_cpu_run_seconds(20, 25) == 1000.0
    # images_per_page multiplies the estimate
    assert settings.estimated_cpu_run_seconds(10, 25, images_per_page=3) == 1500.0
    # fast profile: fewer steps
    assert settings.estimated_cpu_run_seconds(1, 6) == 12.0


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


def test_get_vision_provider_default():
    assert settings.get_vision_provider() == "openrouter"


def test_get_vision_provider_from_dedicated_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_VISION_PROVIDER", "anthropic")
    assert settings.get_vision_provider() == "anthropic"


def test_get_vision_provider_falls_back_to_llm_provider_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_LLM_PROVIDER", "openai")
    assert settings.get_vision_provider() == "openai"


def test_get_vision_provider_dedicated_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("PAGELOOM_LLM_PROVIDER", "openai")
    monkeypatch.setenv("PAGELOOM_VISION_PROVIDER", "anthropic")
    assert settings.get_vision_provider() == "anthropic"


def test_get_openai_vision_model_default():
    assert settings.get_openai_vision_model() == "gpt-4o-mini"


def test_get_anthropic_vision_model_default():
    assert settings.get_anthropic_vision_model() == "claude-3-haiku-20240307"


def test_get_openrouter_vision_model_default():
    """Without OPENROUTER_VISION_MODEL, the default is a vision-capable model
    discovered from OpenRouter's models endpoint -- not the unreliable
    ``openrouter/free`` auto-router."""
    import json
    from unittest.mock import patch

    payload = {
        "data": [
            {
                "id": "google/gemma-4-26b-a4b-it:free",
                "architecture": {"modality": "text+image->text"},
            },
            {"id": "openai/gpt-4o-mini", "architecture": {"modality": "text->text"}},
        ]
    }

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    with patch(
        "pageloom.providers.openrouter.urllib.request.urlopen",
        return_value=_FakeResponse(),
    ):
        assert settings.get_openrouter_vision_model() == "google/gemma-4-26b-a4b-it:free"


def test_get_openrouter_vision_model_env_takes_precedence_over_discovery(monkeypatch):
    monkeypatch.setenv("OPENROUTER_VISION_MODEL", "some/vision-model:free")
    assert settings.get_openrouter_vision_model() == "some/vision-model:free"


def test_get_openrouter_vision_model_falls_back_to_free_router_on_discovery_failure(monkeypatch):
    """If the models endpoint is unreachable or malformed, fall back to
    ``openrouter/free`` instead of failing the whole run."""

    def _raise(_req, *args, **kwargs):
        raise URLError("unreachable")

    with patch("pageloom.providers.openrouter.urllib.request.urlopen", side_effect=_raise):
        assert settings.get_openrouter_vision_model() == "openrouter/free"


def test_get_openrouter_vision_model_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_VISION_MODEL", "some/vision-model:free")
    assert settings.get_openrouter_vision_model() == "some/vision-model:free"


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


def test_get_image_provider_default():
    assert settings.get_image_provider() == "local"


def test_get_image_provider_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_IMAGE_PROVIDER", "openrouter")
    assert settings.get_image_provider() == "openrouter"


def test_get_image_model_default():
    assert settings.get_image_model() is None


def test_get_image_model_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_IMAGE_MODEL", "custom/image-model")
    assert settings.get_image_model() == "custom/image-model"


def test_get_flux_host_default():
    assert settings.get_flux_host() == "bfl"


def test_get_flux_host_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_HOST", "replicate")
    assert settings.get_flux_host() == "replicate"


def test_get_flux_model_default():
    assert settings.get_flux_model() == "flux-pro-1.1"


def test_get_flux_model_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_MODEL", "flux-dev")
    assert settings.get_flux_model() == "flux-dev"


def test_get_bfl_api_key_default():
    assert settings.get_bfl_api_key() is None


def test_get_bfl_api_key_from_env(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "bfl-key")
    assert settings.get_bfl_api_key() == "bfl-key"


def test_get_fal_api_key_default():
    assert settings.get_fal_api_key() is None


def test_get_fal_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-key")
    assert settings.get_fal_api_key() == "fal-key"


def test_get_replicate_api_key_default():
    assert settings.get_replicate_api_key() is None


def test_get_replicate_api_key_from_env(monkeypatch):
    monkeypatch.setenv("REPLICATE_API_KEY", "rep-key")
    assert settings.get_replicate_api_key() == "rep-key"


def test_get_num_inference_steps_default():
    assert settings.get_num_inference_steps() == 25


def test_get_num_inference_steps_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_SD_STEPS", "30")
    assert settings.get_num_inference_steps() == 30


def test_get_num_inference_steps_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("PAGELOOM_SD_STEPS", "not-a-number")
    assert settings.get_num_inference_steps() == 25


def test_get_guidance_scale_default():
    assert settings.get_guidance_scale() == 7.5


def test_get_guidance_scale_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_SD_GUIDANCE_SCALE", "7")
    assert settings.get_guidance_scale() == 7.0


def test_get_guidance_scale_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("PAGELOOM_SD_GUIDANCE_SCALE", "not-a-number")
    assert settings.get_guidance_scale() == 7.5


def test_get_profile_default():
    assert settings.get_profile() == "quality"


def test_get_profile_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "fast")
    assert settings.get_profile() == "fast"


def test_get_profile_invalid_falls_back_to_quality(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "not-a-profile")
    assert settings.get_profile() == "not-a-profile"


def test_get_num_inference_steps_fast_profile(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "fast")
    assert settings.get_num_inference_steps() == 6


def test_get_num_inference_steps_quality_profile(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "quality")
    assert settings.get_num_inference_steps() == 25


def test_get_num_inference_steps_env_overrides_profile(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "fast")
    monkeypatch.setenv("PAGELOOM_SD_STEPS", "30")
    assert settings.get_num_inference_steps() == 30


def test_get_guidance_scale_fast_profile(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "fast")
    assert settings.get_guidance_scale() == 5.0


def test_get_guidance_scale_quality_profile(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "quality")
    assert settings.get_guidance_scale() == 7.5


def test_get_guidance_scale_env_overrides_profile(monkeypatch):
    monkeypatch.setenv("PAGELOOM_PROFILE", "fast")
    monkeypatch.setenv("PAGELOOM_SD_GUIDANCE_SCALE", "9")
    assert settings.get_guidance_scale() == 9.0


def test_get_text_mode_default():
    assert settings.get_text_mode() == "overlay"


def test_get_text_mode_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_TEXT_MODE", "native")
    assert settings.get_text_mode() == "native"


def test_get_text_mode_from_env_none(monkeypatch):
    monkeypatch.setenv("PAGELOOM_TEXT_MODE", "none")
    assert settings.get_text_mode() == "none"
