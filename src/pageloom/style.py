"""Style-prompt template builder for consistent illustration prompts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pageloom.style_extractor import VisualStyleExtractor


class StylePromptBuilder:
    """Build consistent Stable Diffusion prompts from an example page."""

    def __init__(
        self,
        example_description: str = "",
        example_image: str | Path | Image.Image | None = None,
        example_style: dict | None = None,
    ) -> None:
        self.example_description = example_description.strip()
        self.example_style: dict | None = None
        if example_image is not None and example_style is not None:
            raise ValueError("Provide either example_image or example_style, not both.")
        if example_image is not None:
            extractor = VisualStyleExtractor(example_image)
            self.example_style = extractor.extract()
            self.example_image = extractor.image
        elif example_style is not None:
            self.example_style = example_style
        else:
            self.example_image = None

    def template(self) -> str:
        """Return a prompt template that embeds the example style description."""
        base = f"A children's book illustration in the style of: {self.example_description}"
        return f"{base}, content: {{content}}."

    def build(self, content: str = "", theme: str = "") -> str:
        """Produce a concrete prompt using the example style and content."""
        parts: list[str] = [
            "A children's book illustration",
        ]
        if theme:
            parts.append(f"theme: {theme}")
        if self.example_description:
            parts.append(f"style reference: {self.example_description}")
        elif self.example_style is not None:
            parts.append(
                f"style reference: {self.example_style.get('composition', {}).get('layout_type', 'inferred layout')}"
            )
        if content:
            parts.append(f"content: {content}")
        return ", ".join(parts) + "."
