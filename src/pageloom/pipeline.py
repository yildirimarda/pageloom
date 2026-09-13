"""Main pipeline orchestration that infers layout and style directly from an example page."""

from __future__ import annotations

import hashlib
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
        force: bool = False,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """Generate matching pages for each content item and compile to PDF."""
        themes = themes or [""] * len(contents)
        pages: list[Image.Image] = []

        if cache_dir is None:
            output_path_obj = Path(output_path) if output_path else Path("output.pdf")
            cache_dir = output_path_obj.parent / ".pageloom_raw"
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        for content, theme in zip(contents, themes):
            prompt = self.prompt_builder.build(
                content=content, theme=theme, page_format=page_format
            )
            width = self.style.get("composition", {}).get("page_dimensions", {}).get("width", 612)
            height = self.style.get("composition", {}).get("page_dimensions", {}).get("height", 612)

            cache_key = f"{content}:{theme}:{page_format}:{int(width)}:{int(height)}:{prompt}"
            cache_filename = hashlib.md5(cache_key.encode("utf-8")).hexdigest() + ".png"
            cache_path = cache_dir / cache_filename

            if not force and cache_path.exists():
                image = Image.open(str(cache_path))
            else:
                image = self.generator.generate(
                    prompt=prompt,
                    width=int(width),
                    height=int(height),
                )
                image.save(str(cache_path), "PNG")

            overlaid = self.overlay.apply(image, text=content, page_format=page_format)
            pages.append(overlaid)
        return self.compiler.compile(pages, output_path)
