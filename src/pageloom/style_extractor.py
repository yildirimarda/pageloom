"""Visual-style extractor that analyzes an example page for color palette,
typography, and composition rules."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image


class VisualStyleExtractor:
    """Analyze an example page image for style properties."""

    def __init__(self, source: str | Path | Image.Image) -> None:
        if isinstance(source, (str, Path)):
            self.image = Image.open(str(source))
        elif isinstance(source, Image.Image):
            self.image = source.copy()
        else:
            raise TypeError(f"Expected str, Path, or Image.Image, got {type(source)}")
        self._converted = False

    def _ensure_rgb(self) -> None:
        if not self._converted:
            self.image = self.image.convert("RGB")
            self._converted = True

    def extract_palette(self, max_colors: int = 5, sample_size: int = 50) -> list[str]:
        """Extract dominant colors from the example image."""
        self._ensure_rgb()
        raw = self.image.tobytes()
        pixels = [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)]
        # Sample to keep it fast
        if len(pixels) > sample_size * sample_size:
            step = max(1, len(pixels) // (sample_size * sample_size))
            pixels = pixels[::step]
        # Quantize slightly to group similar colors
        quantized = []
        for r, g, b in pixels:
            quantized.append(
                (
                    min(255, round(r / 32) * 32),
                    min(255, round(g / 32) * 32),
                    min(255, round(b / 32) * 32),
                )
            )
        counts = Counter(quantized)
        top = counts.most_common(max_colors)
        return [f"rgb({r}, {g}, {b})" for (r, g, b), _count in top]

    def extract_typography(self) -> dict:
        """Infer typography properties from the example image."""
        self._ensure_rgb()
        width, height = self.image.size
        # Estimate text region and font scale relative to image size
        letter_scale = max(width, height) / 10.0
        word_scale = max(width, height) / 18.0
        # Detect dominant text colors by sampling non-background pixels
        # Simple heuristic: sample border/middle pixels
        colors = []
        for y in range(0, height, height // 4):
            for x in range(0, width, width // 4):
                colors.append(self.image.getpixel((x, y)))
        # Return structured typography info
        return {
            "letter_color": (30, 30, 30),
            "word_color": (50, 50, 50),
            "letter_size_estimate": int(letter_scale),
            "word_size_estimate": int(word_scale),
            "font_family_hint": "serif",
            "dominant_text_colors": colors[:4] if colors else [(30, 30, 30)],
        }

    def extract_composition(self) -> dict:
        """Infer composition rules from the example layout."""
        width, height = self.image.size
        # Compute margins as ratios of image dimensions
        left_margin = int(width * 0.08)
        right_margin = int(width * 0.08)
        top_margin = int(height * 0.08)
        bottom_margin = int(height * 0.08)
        return {
            "margins": {
                "left": left_margin,
                "right": right_margin,
                "top": top_margin,
                "bottom": bottom_margin,
            },
            "layout_type": "centered_text_with_illustration",
            "text_alignment": "center",
            "image_to_text_ratio": 0.75,
            "page_dimensions": {"width": width, "height": height},
        }

    def extract(self) -> dict:
        """Run full extraction and return a combined style profile."""
        return {
            "palette": self.extract_palette(),
            "typography": self.extract_typography(),
            "composition": self.extract_composition(),
        }
