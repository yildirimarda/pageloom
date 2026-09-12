"""Pillow-based text overlay layer for letter and word rendering."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


class TextOverlay:
    """Overlay letters and words onto generated illustrations."""

    def __init__(
        self,
        font_path: str | None = None,
        letter_size: int = 120,
        word_size: int = 48,
        letter_color: tuple[int, int, int] = (30, 30, 30),
        word_color: tuple[int, int, int] = (50, 50, 50),
        letter_position: tuple[int, int] | None = None,
        word_position: tuple[int, int] | None = None,
    ) -> None:
        self.font_path = font_path
        self.letter_size = letter_size
        self.word_size = word_size
        self.letter_color = letter_color
        self.word_color = word_color
        self.letter_position = letter_position
        self.word_position = word_position
        self._letter_font = None
        self._word_font = None

    def _get_font(self, size: int):
        try:
            if self.font_path:
                return ImageFont.truetype(self.font_path, size)
        except OSError:
            pass
        # Default font doesn't support custom sizing; ignore size for fallback
        return ImageFont.load_default()

    def apply(
        self,
        image: Image.Image,
        letter: str = "",
        word: str = "",
    ) -> Image.Image:
        overlay_image = image.copy()
        draw = ImageDraw.Draw(overlay_image)

        if letter:
            font = self._get_font(self.letter_size)
            x = self.letter_position[0] if self.letter_position else 40
            y = self.letter_position[1] if self.letter_position else 40
            draw.text((x, y), letter.upper(), fill=self.letter_color, font=font)

        if word:
            font = self._get_font(self.word_size)
            x = self.word_position[0] if self.word_position else 40
            y = self.word_position[1] if self.word_position else 180
            draw.text((x, y), word.lower(), fill=self.word_color, font=font)

        return overlay_image
