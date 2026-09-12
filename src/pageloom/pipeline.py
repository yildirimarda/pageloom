"""Main pipeline orchestration that infers layout and style directly from an example page."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pageloom.generator import IllustrationGenerator
from pageloom.overlay import TextOverlay
from pageloom.pdf_compiler import PDFCompiler
from pageloom.style import StylePromptBuilder
from pageloom.style_extractor import VisualStyleExtractor


class Pipeline:
    """Generate pages by inferring style and layout from an example page."""

    def __init__(
        self,
        example_image: str | Path | Image.Image,
        generator: IllustrationGenerator | None = None,
        compiler: PDFCompiler | None = None,
    ) -> None:
        self.extractor = VisualStyleExtractor(example_image)
        self.style = self.extractor.extract()
        self.prompt_builder = StylePromptBuilder(
            example_style=self.style,
        )
        self.generator = generator or IllustrationGenerator()
        self.overlay = TextOverlay(style_profile=self.style)
        self.compiler = compiler or PDFCompiler()

    def run(
        self,
        contents: list[str],
        themes: list[str] | None = None,
        output_path: str | Path = "output.pdf",
        page_format: str = "standard",
    ) -> Path:
        """Generate matching pages for each content item and compile to PDF."""
        themes = themes or [""] * len(contents)
        pages: list[Image.Image] = []
        for content, theme in zip(contents, themes):
            prompt = self.prompt_builder.build(
                content=content, theme=theme, page_format=page_format
            )
            width = self.style.get("composition", {}).get("page_dimensions", {}).get("width", 612)
            height = self.style.get("composition", {}).get("page_dimensions", {}).get("height", 612)
            image = self.generator.generate(
                prompt=prompt,
                width=int(width),
                height=int(height),
            )
            overlaid = self.overlay.apply(image, text=content, page_format=page_format)
            pages.append(overlaid)
        return self.compiler.compile(pages, output_path)
