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
        base = f"An illustration in the style of: {self.example_description}"
        return f"{base}, content: {{content}}."

    def build(
        self,
        content: str = "",
        theme: str = "",
        page_format: str = "standard",
        element_count: int | None = None,
        arrangement: str | None = None,
        text_image_relationship: str | None = None,
        use_reference_style: bool = False,
    ) -> str:
        """Produce a concrete prompt using the example style and content.

        When ``use_reference_style`` is true (the provider accepts reference
        images and at least one reference is being sent), the prompt instructs
        the model to match the style of the reference images instead of
        embedding the text-only style description.
        """
        parts: list[str] = []
        if page_format == "object_name":
            parts.append("Photo-realistic illustration")
            if content:
                parts.append(f"object name: {content}")
        elif page_format == "themed_scene":
            parts.append("Themed scene illustration")
            if theme:
                parts.append(f"theme: {theme}")
            if content:
                parts.append(f"scene: {content}")
        else:
            parts.append("An illustration")
            if theme:
                parts.append(f"theme: {theme}")
            if content:
                parts.append(f"content: {content}")
        structure_parts: list[str] = []
        if element_count is not None:
            structure_parts.append(f"{element_count} illustrated element(s)")
        if arrangement is not None:
            structure_parts.append(f"arranged in a {arrangement} layout")
        if text_image_relationship is not None:
            if text_image_relationship == "label_per_image":
                structure_parts.append("each image paired with its label")
            elif text_image_relationship == "multiple_labels":
                structure_parts.append("multiple text labels")
            elif text_image_relationship == "no_text":
                structure_parts.append("no text")
            else:
                structure_parts.append(f"text-image relationship: {text_image_relationship}")
        if structure_parts:
            parts.append("Structure: " + ", ".join(structure_parts))
        if use_reference_style:
            parts.append("match the style of the reference images")
        elif self.example_description:
            parts.append(f"style reference: {self.example_description}")
        elif self.example_style is not None:
            style = self.example_style.get("style", {}) or {}
            if not isinstance(style, dict):
                style = {}
            if style:
                # Build the diffusion prompt from the example's real style
                # (medium, line weight, palette in colour names, background,
                # lighting) plus the extracted palette, instead of the layout
                # label inferred from the example's composition.
                medium = style.get("medium", "")
                line_weight = style.get("line_weight", "")
                background = style.get("background", "")
                lighting = style.get("lighting", "")
                style_bits = [
                    bit
                    for bit in [
                        medium,
                        f"{line_weight} line weight" if line_weight else "",
                        f"{background} background" if background else "",
                        f"{lighting} lighting" if lighting else "",
                    ]
                    if bit
                ]
                vision_palette = style.get("palette", [])
                if not isinstance(vision_palette, list):
                    vision_palette = []
                extracted_palette = self.example_style.get("palette", [])
                if not isinstance(extracted_palette, list):
                    extracted_palette = []
                palette_names = [str(p) for p in vision_palette + extracted_palette if p]
                parts.append(
                    "style reference: "
                    + ", ".join(style_bits)
                    + "; palette: "
                    + ", ".join(palette_names)
                )
            else:
                # No vision style description available: fall back to the
                # layout label inferred from the example's composition.
                parts.append(
                    f"style reference: {self.example_style.get('composition', {}).get('layout_type', 'inferred layout')}"
                )
        return ", ".join(parts) + "."
