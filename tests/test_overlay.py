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
