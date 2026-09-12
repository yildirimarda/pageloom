"""Pillow-based text overlay layer for content rendering."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


class TextOverlay:
    """Overlay text onto generated illustrations based on an inferred style profile."""

    def __init__(
        self,
        font_path: str | None = None,
        style_profile: dict | None = None,
        text_size: int = 120,
        text_color: tuple[int, int, int] = (30, 30, 30),
        text_position: tuple[int, int] | None = None,
    ) -> None:
        self.font_path = font_path
        self.style_profile = style_profile or {}
        # Infer typography from profile if available
        typo = self.style_profile.get("typography", {})
        self.text_size = text_size
        if typo.get("letter_size_estimate"):
            self.text_size = int(typo.get("letter_size_estimate", text_size))
        else:
            # Fallback to estimated scale from composition
            comp = self.style_profile.get("composition", {})
            dims = comp.get("page_dimensions", {})
            if dims.get("width") and dims.get("height"):
                max_dim = max(dims.get("width", 512), dims.get("height", 512))
                self.text_size = int(max_dim / 8.0)
        self.text_color = text_color
        if typo.get("letter_color"):
            self.text_color = tuple(typo.get("letter_color", text_color))
        elif typo.get("word_color"):
            self.text_color = tuple(typo.get("word_color", text_color))
        # Infer position from composition margins
        comp = self.style_profile.get("composition", {})
        margins = comp.get("margins", {})
        dims = comp.get("page_dimensions", {})
        width = dims.get("width", 512)
        height = dims.get("height", 512)
        if text_position is not None:
            self.text_position = text_position
        else:
            left_margin = margins.get("left", int(width * 0.08))
            top_margin = margins.get("top", int(height * 0.08))
            self.text_position = (left_margin + 20, top_margin + 20)
        self._font = None

    def _get_font(self, size: int):
        try:
            if self.font_path:
                return ImageFont.truetype(self.font_path, size)
        except OSError:
            pass
        return ImageFont.load_default()

    def apply(
        self,
        image: Image.Image,
        text: str = "",
    ) -> Image.Image:
        overlay_image = image.copy()
        draw = ImageDraw.Draw(overlay_image)

        if text:
            font = self._get_font(self.text_size)
            x, y = self.text_position
            draw.text((x, y), text, fill=self.text_color, font=font)

        return overlay_image
