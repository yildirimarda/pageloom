import importlib.util
import pathlib

PACKAGE_DIR = pathlib.Path("src/pageloom")
MODULE_STUBS = [
    "generator",
    "overlay",
    "pipeline",
    "pdf_compiler",
    "style",
    "style_extractor",
    "consistency",
]


def test_package_has_init():
    init_path = PACKAGE_DIR / "__init__.py"
    assert init_path.exists()


def test_all_module_stubs_exist():
    for name in MODULE_STUBS:
        stub_path = PACKAGE_DIR / f"{name}.py"
        assert stub_path.exists(), f"missing stub: {name}.py"


def test_all_modules_importable():
    import sys

    sys.path.insert(0, "src")
    for name in MODULE_STUBS:
        module = importlib.import_module(f"pageloom.{name}")
        assert module is not None
