def test_readme_exists_and_has_kdp_summary():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md should exist"
    content = readme_path.read_text()
    assert "PageLoom" in content
    assert "KDP" in content
    assert "pipeline" in content.lower()
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
