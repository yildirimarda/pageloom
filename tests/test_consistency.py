from pathlib import Path

from PIL import Image

from pageloom.consistency import ConsistencyGuard


def test_guard_init_with_image(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="blue")
    img.save(img_path)
    guard = ConsistencyGuard(example_image=str(img_path))
    assert guard.example_style is not None
    assert guard.example_image is not None


def test_guard_init_with_style():
    style = {
        "palette": ["rgb(0, 0, 255)"],
        "typography": {
            "letter_size_estimate": 50,
            "word_size_estimate": 30,
            "font_family_hint": "serif",
        },
        "composition": {
            "layout_type": "centered_text_with_illustration",
            "page_dimensions": {"width": 512, "height": 512},
        },
    }
    guard = ConsistencyGuard(example_style=style)
    assert guard.example_style == style


def test_guard_init_neither_raises():
    try:
        ConsistencyGuard()
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_guard_init_both_raises():
    style = {"palette": [], "typography": {}, "composition": {}}
    try:
        ConsistencyGuard(example_image=Image.new("RGB", (100, 100)), example_style=style)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_guard_verify_matching_image():
    img = Image.new("RGB", (400, 400), color="purple")
    guard = ConsistencyGuard(example_image=img)
    result = guard.verify(img)
    assert isinstance(result, dict)
    assert "consistent" in result
    assert "issues" in result
    assert "details" in result
    assert result["consistent"] is True
    assert result["issues"] == []


def test_guard_verify_different_image():
    img = Image.new("RGB", (400, 400), color="red")
    guard = ConsistencyGuard(example_image=img)
    different = Image.new("RGB", (200, 200), color="green")
    result = guard.verify(different)
    assert result["consistent"] is False
    assert len(result["issues"]) > 0
