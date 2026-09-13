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

    def _label_positions(
        self,
        width: int,
        height: int,
        count: int,
        arrangement: str | None = None,
    ) -> list[tuple[int, int]]:
        """Compute positions for multiple labels based on arrangement."""
        if arrangement == "grid":
            cols = max(1, int(count**0.5))
            rows = max(1, (count + cols - 1) // cols)
            cell_w = width // cols
            cell_h = height // rows
            positions = []
            for i in range(count):
                r, c = i // cols, i % cols
                x = c * cell_w + cell_w // 4
                y = r * cell_h + cell_h // 4
                positions.append((x, y))
            return positions
        elif arrangement == "row":
            step = width // max(1, count)
            y = height // 2
            return [(i * step + step // 4, y) for i in range(count)]
        elif arrangement == "column":
            step = height // max(1, count)
            x = width // 4
            return [(x, i * step + step // 4) for i in range(count)]
        else:
            # Centered: spread evenly horizontally near bottom
            step = max(50, width // max(1, count))
            y = height - int(height * 0.15)
            return [(i * step + step // 4, y) for i in range(count)]

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
        page_format: str = "standard",
        text_image_relationship: str | None = None,
        element_count: int | None = None,
        arrangement: str | None = None,
    ) -> Image.Image:
        overlay_image = image.copy()
        # Design decision: `no_text` suppresses all text overlay; `multiple_labels`
        # splits content by commas and places each label individually based on
        # arrangement; full multi-sub-image compositing remains future work.
        if text_image_relationship == "no_text":
            return overlay_image

        draw = ImageDraw.Draw(overlay_image)

        if text_image_relationship == "multiple_labels":
            labels = [label.strip() for label in text.split(",") if label.strip()]
            if not labels:
                labels = [text] if text else []
            # Place labels based on arrangement and element count
            count = element_count or len(labels) or 1
            positions = self._label_positions(
                image.width, image.height, len(labels) or count, arrangement
            )
            font = self._get_font(self.text_size)
            for label, (lx, ly) in zip(labels, positions):
                draw.text((lx, ly), label, fill=self.text_color, font=font)
            return overlay_image

        if text:
            font = self._get_font(self.text_size)
            if page_format == "object_name":
                # Center the object name prominently near the bottom
                x = (image.width - 100) // 2
                y = image.height - int(image.height * 0.15)
                draw.text((x, y), text, fill=self.text_color, font=font)
            elif page_format == "themed_scene":
                # Smaller label centered horizontally near the bottom
                x = (image.width - 150) // 2
                y = image.height - int(image.height * 0.1)
                draw.text((x, y), text, fill=self.text_color, font=font)
            else:
                x, y = self.text_position
                draw.text((x, y), text, fill=self.text_color, font=font)

        return overlay_image
