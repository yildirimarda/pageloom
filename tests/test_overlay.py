from PIL import Image

from pageloom.overlay import TextOverlay


def test_text_overlay_init_defaults():
    overlay = TextOverlay()
    assert overlay.text_size > 0
    assert overlay.text_color == (30, 30, 30)


def test_text_overlay_apply_text():
    base = Image.new("RGB", (400, 400), color="white")
    overlay = TextOverlay()
    result = overlay.apply(base, text="apple")
    assert isinstance(result, Image.Image)
    assert result.size == (400, 400)
    assert result != base


def test_text_overlay_apply_empty_text():
    base = Image.new("RGB", (300, 300), color="blue")
    overlay = TextOverlay()
    result = overlay.apply(base, text="")
    assert result.size == (300, 300)


def test_text_overlay_apply_object_name_format():
    base = Image.new("RGB", (400, 400), color="white")
    overlay = TextOverlay()
    result = overlay.apply(base, text="apple", page_format="object_name")
    assert isinstance(result, Image.Image)
    assert result.size == (400, 400)


def test_text_overlay_apply_themed_scene_format():
    base = Image.new("RGB", (500, 500), color="blue")
    overlay = TextOverlay()
    result = overlay.apply(base, text="autumn", page_format="themed_scene")
    assert isinstance(result, Image.Image)
    assert result.size == (500, 500)


def test_text_overlay_custom_font_path_fallback(tmp_path):
    overlay = TextOverlay(font_path=str(tmp_path / "nonexistent.ttf"))
    base = Image.new("RGB", (200, 200), color="green")
    result = overlay.apply(base, text="test")
    assert isinstance(result, Image.Image)
    assert result.size == (200, 200)


def test_text_overlay_custom_text_position():
    overlay = TextOverlay(text_position=(10, 10))
    base = Image.new("RGB", (300, 300), color="red")
    result = overlay.apply(base, text="label")
    assert result.size == (300, 300)
    assert overlay.text_position == (10, 10)


def test_text_overlay_apply_returns_new_image():
    base = Image.new("RGB", (400, 400), color="yellow")
    overlay = TextOverlay()
    result = overlay.apply(base, text="hello")
    assert result is not base


def test_text_overlay_with_style_profile():
    base = Image.new("RGB", (512, 512), color="yellow")
    profile = {
        "typography": {
            "letter_size_estimate": 80,
            "word_color": (20, 20, 20),
        },
        "composition": {
            "margins": {"left": 50, "top": 60},
            "page_dimensions": {"width": 512, "height": 512},
        },
    }
    overlay = TextOverlay(style_profile=profile)
    result = overlay.apply(base, text="cat")
    assert result.size == (512, 512)
