import tomllib


def test_pyproject_toml_has_dependencies():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    names = {d.split(">")[0].split("=")[0].strip().lower() for d in deps}
    assert "diffusers" in names
    assert "pillow" in names
    assert "pypdf" in names


def test_pyproject_toml_has_dev_dependencies():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    dev = data["project"]["optional-dependencies"]["dev"]
    dev_names = {d.split("=")[0].strip().lower() for d in dev}
    assert "pytest" in dev_names
    assert "ruff" in dev_names


def test_pyproject_toml_has_backend_notes():
    with open("pyproject.toml", "r") as f:
        content = f.read()
    assert "MPS" in content
    assert "CPU" in content
    assert "CUDA" in content
