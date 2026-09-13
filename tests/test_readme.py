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
