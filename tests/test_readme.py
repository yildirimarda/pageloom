def test_readme_exists_and_has_summary():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md should exist"
    content = readme_path.read_text()
    assert "PageLoom" in content
    assert "pipeline" in content.lower()
    assert "example" in content.lower()
    lines = content.splitlines()
    assert len(lines) >= 2


def test_readme_has_installation_sections():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "Installation" in content
    assert "Docker" in content
    assert "local" in content.lower() or "Local" in content


def test_readme_has_model_and_dependency_setup():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "Model" in content or "model" in content.lower()
    assert "Dependency" in content or "dependency" in content.lower()
    assert "diffusers" in content.lower() or "Stable Diffusion" in content


def test_readme_has_cli_usage_examples():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "CLI" in content or "Usage" in content
    assert "python" in content.lower()


def test_readme_has_output_structure_and_troubleshooting():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "Output" in content or "output" in content.lower()
    assert "Troubleshooting" in content or "troubleshooting" in content.lower()


def test_readme_documents_llm_provider_configuration():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "LLM Provider" in content or "LLM provider" in content
    assert "PAGELOOM_LLM_PROVIDER" in content
    assert "OPENAI_API_KEY" in content
    assert "ANTHROPIC_API_KEY" in content
    assert "OPENROUTER_API_KEY" in content


def test_readme_documents_llm_provider_flag():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "--llm-provider" in content
    assert "--use-llm" in content


def test_readme_documents_supported_page_structures():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "single_letter_word_illustration" in content
    assert "object_name_pair" in content
    assert "multi_item_scene" in content


def test_readme_documents_adding_new_structure():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "Adding a New" in content or "new structure" in content.lower()
    assert "PageStructureClassifier" in content


def test_readme_documents_automatic_vision_analysis():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "vision" in content.lower()
    assert "--no-vision" in content
    assert "--vision-provider" in content
    assert "PAGELOOM_VISION_PROVIDER" in content
    assert "OPENROUTER_VISION_MODEL" in content
    # The doc must explain why Stable Diffusion can't be reused for analysis.
    assert "generation" in content.lower() and "Stable Diffusion" in content


def test_readme_documents_how_to_find_free_vision_models():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    # Must reference the OpenRouter models page with image filter.
    assert "openrouter.ai/models" in content
    assert "image" in content.lower()
    # Must reference the vision-models collection.
    assert "vision-models" in content
    # Must document the free auto-router rationale.
    assert "openrouter/free" in content
    assert "auto-router" in content.lower()
    # Must mention observed free vision-capable models with a freshness caveat.
    assert "google/gemma" in content or "gemma" in content.lower()
    assert "free" in content.lower()
    assert "Sept 2026" in content or "changes constantly" in content


def test_readme_documents_consistency_opt_out():
    """The per-page consistency check opt-out must be documented on both the
    CLI side (--no-consistency) and the library side (verify_consistency=False)."""
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "--no-consistency" in content
    assert "verify_consistency=False" in content
    assert "Consistency check: N/M" in content


def test_readme_documents_choosing_an_image_model():
    """The 'Choosing an image model' section must document each provider: how it
    is configured, which accept reference images, rough cost behaviour, why the
    model IDs are settings, and where to look up current models."""
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    # The section itself exists.
    assert "Choosing an image model" in content

    # Provider selection is documented.
    assert "PAGELOOM_IMAGE_PROVIDER" in content
    assert "PAGELOOM_IMAGE_MODEL" in content
    assert "--image-provider" in content
    assert "--image-model" in content

    # Each provider is documented with its credential and default model.
    for provider in ("local", "openai", "gemini", "openrouter", "flux"):
        assert (
            f"`{provider}`" in content
            or f"**{provider}**" in content
            or f"`{provider}` —" in content
        )
    assert "OPENAI_API_KEY" in content
    assert "GEMINI_API_KEY" in content
    assert "OPENROUTER_API_KEY" in content
    assert "BFL_API_KEY" in content or "FAL_KEY" in content or "REPLICATE_API_KEY" in content
    assert "PAGELOOM_MODEL_ID" in content
    assert "OPENAI_IMAGE_MODEL" in content
    assert "GEMINI_IMAGE_MODEL" in content
    assert "OPENROUTER_IMAGE_MODEL" in content
    assert "PAGELOOM_FLUX_MODEL" in content

    # Reference image support (and lack thereof) is documented.
    assert "reference" in content.lower()
    assert "text-to-image only" in content.lower() or "not forwarded" in content.lower()

    # Rough cost behaviour is documented.
    assert "$0.00" in content
    assert "$0.040" in content or "$0.080" in content
    assert "$0.030" in content
    assert "$0.055" in content
    assert "PAGELOOM_IMAGE_PRICE_TABLE" in content
    assert "--max-cost" in content
    assert "--estimate" in content

    # Why the model IDs are settings is explained.
    assert "Why the model IDs are settings" in content
    assert "settings/env" in content.lower() or "settings/env" in content

    # Where to look up current models is documented.
    assert "openrouter.ai/models" in content
    assert "huggingface.co/models" in content
    assert "list_openrouter_image_models" in content


def test_readme_documents_generalized_output_configuration():
    """The README must document the generalized output formats, page-size
    defaults, text modes, and configuration flags that landed with Milestone 11."""
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    # Output formats beyond a single PDF.
    assert "--output-format" in content
    for fmt in ("pdf", "png", "jpeg", "zip"):
        assert fmt in content
    assert "{stem}_{index:03d}" in content or "out_001.png" in content

    # Page size defaults to the example's own aspect ratio; --page-size presets.
    assert "--page-size" in content
    assert "square" in content and "a4" in content and "letter" in content
    assert "WIDTHxHEIGHT" in content
    assert "DEFAULT_PAGE_SIZE" in content
    assert "KDP_PAGE_SIZE" in content  # deprecated alias kept for one release
    assert "example image's own" in content or "example's own" in content

    # Text rendering modes.
    assert "--text-mode" in content
    for mode in ("overlay", "native", "none"):
        assert mode in content

    # Content file (no-LLM content source).
    assert "--content-file" in content
    assert "txt/csv/json" in content

    # Page format derived from detected structure.
    assert "--format" in content
    for page_format in ("standard", "object_name", "themed_scene"):
        assert page_format in content

    # Generation profile and cost controls.
    assert "--profile" in content
    assert "fast" in content and "quality" in content
    assert "--max-cost" in content
    assert "--estimate" in content

    # Generalized environment variables.
    for env_var in (
        "PAGELOOM_TEXT_MODE",
        "PAGELOOM_PROFILE",
        "PAGELOOM_CPU_SLOW_WARN_SECONDS",
        "PAGELOOM_QUALITY_GATE_MAX_RETRIES",
        "PAGELOOM_IMAGE_STRENGTH",
        "PAGELOOM_SD_STEPS",
        "PAGELOOM_SD_GUIDANCE_SCALE",
    ):
        assert env_var in content

    # All image providers are selectable (not just ``local``).
    for provider in ("local", "openai", "gemini", "openrouter", "flux"):
        assert provider in content

    # Vision-derived structural relationships beyond the three heuristic buckets.
    assert "multiple_labels" in content
    assert "no_text" in content

    # Generalization language: no children's-book assumptions, any page type.
    assert "no children's-book assumptions" in content or "no children" in content.lower()
    assert "any page type" in content.lower() or "any page" in content.lower()
