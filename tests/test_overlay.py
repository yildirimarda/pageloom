from PIL import Image

from pageloom.overlay import TextOverlay


def test_text_overlay_init_defaults():
    overlay = TextOverlay()
    assert overlay.letter_size == 120
    assert overlay.word_size == 48
    assert overlay.letter_color == (30, 30, 30)


def test_text_overlay_apply_letter_and_word():
    base = Image.new("RGB", (400, 400), color="white")
    overlay = TextOverlay()
    result = overlay.apply(base, letter="A", word="apple")
    assert isinstance(result, Image.Image)
    assert result.size == (400, 400)
    assert result != base


def test_text_overlay_apply_only_letter():
    base = Image.new("RGB", (300, 300), color="blue")
    overlay = TextOverlay()
    result = overlay.apply(base, letter="B")
    assert result.size == (300, 300)


def test_text_overlay_apply_only_word():
    base = Image.new("RGB", (300, 300), color="green")
    overlay = TextOverlay()
    result = overlay.apply(base, word="ball")
    assert result.size == (300, 300)


def test_text_overlay_custom_positions():
    base = Image.new("RGB", (512, 512), color="yellow")
    overlay = TextOverlay(
        letter_position=(100, 100),
        word_position=(100, 250),
    )
    result = overlay.apply(base, letter="C", word="cat")
    assert result.size == (512, 512)
