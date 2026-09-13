"""Tests that Docker caching / HF_HOME volume setup is documented and handled."""

import pathlib

DOCKERFILE = pathlib.Path("Dockerfile")
DOCKERIGNORE = pathlib.Path(".dockerignore")


def test_dockerfile_sets_hf_home():
    content = DOCKERFILE.read_text()
    assert "HF_HOME=/root/.cache/huggingface" in content


def test_dockerfile_creates_hf_cache_dir():
    content = DOCKERFILE.read_text()
    assert "mkdir -p /root/.cache/huggingface" in content


def test_dockerfile_declares_hf_volume():
    content = DOCKERFILE.read_text()
    assert 'VOLUME ["/root/.cache/huggingface"]' in content


def test_dockerignore_exists():
    assert DOCKERIGNORE.exists()


def test_dockerignore_excludes_cache_dirs():
    content = DOCKERIGNORE.read_text()
    assert "huggingface/" in content
    assert ".cache/" in content


def test_settings_uses_hf_home_when_set(monkeypatch):
    from pageloom import settings

    monkeypatch.setenv("HF_HOME", "/persisted/hf_home")
    assert settings.get_hf_cache_dir() == "/persisted/hf_home"
