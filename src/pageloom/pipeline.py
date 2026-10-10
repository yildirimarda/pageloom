"""Main pipeline orchestration that infers layout and style directly from an example page."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

from pageloom import settings
from pageloom.consistency import ConsistencyGuard
from pageloom.generator import IllustrationGenerator
from pageloom.overlay import TextOverlay
from pageloom.page_structure_classifier import detect_structure
from pageloom.pdf_compiler import PDFCompiler
from pageloom.style import StylePromptBuilder
from pageloom.style_extractor import VisualStyleExtractor


def round_to_multiple(dim: int, base: int = 8) -> int:
    """Round a dimension up to the nearest multiple of ``base``.

    Stable Diffusion's VAE requires both generation dimensions to be multiples
    of 8, or ``check_inputs()`` raises ``ValueError``. Example-page dimensions
    come from the source image's raw pixel size and are not guaranteed to
    satisfy that constraint, so every dimension passed to
    ``IllustrationGenerator.generate()`` must be rounded first. Rounds up by at
    most ``base - 1`` pixels, keeping the page close to the example's size and
    aspect ratio.
    """
    if base <= 0:
        raise ValueError("base must be positive")
    return ((int(dim) + base - 1) // base) * base


class Pipeline:
    """Generate pages by inferring style and layout from an example page."""

    def __init__(
        self,
        example_image: str | Path | Image.Image,
        description: str = "",
        generator: IllustrationGenerator | None = None,
        compiler: PDFCompiler | None = None,
        page_structure: dict | None = None,
        vision_analyzer: object | None = None,
        verify_consistency: bool = True,
    ) -> None:
        self.extractor = VisualStyleExtractor(example_image, description=description)
        self.style = self.extractor.extract()
        self.prompt_builder = StylePromptBuilder(
            example_style=self.style,
        )
        self.generator = generator or IllustrationGenerator()
        self.overlay = TextOverlay(style_profile=self.style)
        self.compiler = compiler or PDFCompiler()
        # Detect page structure for full structural routing through the pipeline.
        # A pre-computed `page_structure` (e.g. one the CLI already obtained via
        # vision analysis) is used as-is to avoid a second, redundant analysis
        # call; otherwise detection runs now, preferring `vision_analyzer` when
        # given and falling back to the heuristic classifier automatically.
        self.page_structure = page_structure or detect_structure(
            source=example_image, description=description, vision_analyzer=vision_analyzer
        )
        # Consistency verification runs on every generated page by default
        # (opt out with `verify_consistency=False`). The guard is
        # built from the style/structure already extracted above (no duplicate
        # analysis of the example) and receives the same `vision_analyzer` used
        # for the example so generated pages are re-classified with the same
        # method (vision or heuristic) for a like-for-like structural check.
        # With vision enabled that costs one extra vision API call per page on
        # a first run (the raw-illustration and vision caches only help on
        # reruns), and OpenRouter's free tier is rate-capped -- hence the opt-out.
        self.verify_consistency = verify_consistency
        self.vision_analyzer = vision_analyzer
        self.consistency_guard = ConsistencyGuard(
            example_style=self.style,
            page_structure=self.page_structure,
            description=description,
            vision_analyzer=vision_analyzer,
        )
        # One result dict per page from the most recent run() (see _verify_page).
        self.consistency_results: list[dict] = []

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
        # Derive page format from detected structure when user hasn't overridden it
        # with a non-default value. The CLI passes explicit formats; the library
        # can infer from the example page structure.
        derived_format = page_format
        if page_format == "standard" and self.page_structure:
            pattern = self.page_structure.get("content_pattern", "")
            if pattern == "object_name_pair":
                derived_format = "object_name"
            elif pattern == "multi_item_scene":
                derived_format = "themed_scene"
        page_format = derived_format
        output_path_obj = Path(output_path) if output_path else Path("output.pdf")
        output_dir = settings.get_output_dir()
        if output_dir is not None and not output_path_obj.is_absolute():
            output_path_obj = Path(output_dir) / output_path_obj
        themes = themes or [""] * len(contents)
        pages: list[Image.Image] = []
        # Example-page dimensions are taken verbatim from the source image and
        # are not guaranteed to be multiples of 8, which Stable Diffusion's VAE
        # requires. Round every dimension that reaches the generator so real
        # (non-mocked) runs never hit `check_inputs()`' ValueError.
        width = round_to_multiple(
            self.style.get("composition", {}).get("page_dimensions", {}).get("width", 612)
        )
        height = round_to_multiple(
            self.style.get("composition", {}).get("page_dimensions", {}).get("height", 612)
        )

        if cache_dir is None:
            cache_dir_path = output_path_obj.parent / ".pageloom_raw"
            cache_dir = cache_dir_path
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        for content, theme in zip(contents, themes):
            structure = self.page_structure or {}
            text_rel = structure.get("text_image_relationship")

            # Multi-sub-image compositing for multiple_labels
            if text_rel == "multiple_labels":
                labels = [label.strip() for label in content.split(",") if label.strip()]
                if not labels:
                    labels = [content] if content else [""]
                count = structure.get("element_count")
                if count is None:
                    count = len(labels) or 1
                else:
                    count = max(count, len(labels) or 1)
                # Build prompts and generate sub-images
                sub_images: list[Image.Image] = []
                for i in range(count):
                    label_content = labels[i % len(labels)] if labels else content
                    sub_page_format = page_format
                    sub_prompt = self.prompt_builder.build(
                        content=label_content,
                        theme=theme,
                        page_format=sub_page_format,
                        element_count=1,
                        arrangement=structure.get("arrangement"),
                        text_image_relationship="label_per_image",
                    )
                    # Size each sub-image based on arrangement
                    arrangement = structure.get("arrangement", "centered")
                    if arrangement == "grid":
                        cols = max(1, int(count**0.5))
                        rows = max(1, (count + cols - 1) // cols)
                        sub_w = max(150, int(width) // cols)
                        sub_h = max(150, int(height) // rows)
                    elif arrangement == "row":
                        sub_w = max(150, int(width) // max(1, count))
                        sub_h = max(150, int(height) // 2)
                    elif arrangement == "column":
                        sub_w = max(150, int(width) // 2)
                        sub_h = max(150, int(height) // max(1, count))
                    else:
                        sub_w = max(150, int(width) // max(1, count))
                        sub_h = max(150, int(height) // 2)

                    # Generate sub-image (skip individual sub-caching for simplicity)
                    sub_image = self.generator.generate(
                        prompt=sub_prompt,
                        width=round_to_multiple(sub_w),
                        height=round_to_multiple(sub_h),
                    )
                    sub_images.append(sub_image)

                # Composite sub-images onto a single full-size canvas.
                # Pictures are confined to the upper ~70% so the bottom text band
                # stays clear for the per-label words (no overlapping text).
                composited = Image.new("RGB", (int(width), int(height)), color="white")
                if sub_images:
                    arrangement = structure.get("arrangement", "centered")
                    if arrangement == "grid":
                        cols = max(1, int(len(sub_images) ** 0.5))
                        rows = max(1, (len(sub_images) + cols - 1) // cols)
                    elif arrangement == "row":
                        cols, rows = len(sub_images), 1
                    elif arrangement == "column":
                        cols, rows = 1, len(sub_images)
                    else:
                        cols, rows = len(sub_images), 1

                    label_band_h = max(1, int(int(height) * 0.30))
                    pic_area_h = int(height) - label_band_h
                    cell_w = int(width) // cols if cols > 0 else int(width)
                    cell_h = pic_area_h // rows if rows > 0 else pic_area_h

                    for idx, sub_img in enumerate(sub_images):
                        sub_img = sub_img.convert("RGB")
                        sub_img = sub_img.resize((max(1, cell_w - 20), max(1, cell_h - 20)))
                        if arrangement == "grid":
                            r, c = idx // cols, idx % cols
                            x = c * cell_w + (cell_w - sub_img.width) // 2
                            y = r * cell_h + (cell_h - sub_img.height) // 2
                        elif arrangement == "row":
                            x = idx * cell_w + (cell_w - sub_img.width) // 2
                            y = (pic_area_h - sub_img.height) // 2
                        elif arrangement == "column":
                            x = (int(width) - sub_img.width) // 2
                            y = idx * cell_h + (cell_h - sub_img.height) // 2
                        else:
                            step = int(width) // max(1, len(sub_images))
                            x = idx * step + (step - sub_img.width) // 2
                            y = (pic_area_h - sub_img.height) // 2
                        composited.paste(sub_img, (x, y))

                # Apply text overlay with all labels on the composited canvas
                overlay_text = ", ".join(
                    [labels[i % len(labels)] for i in range(count)] if labels else [content] * count
                )
                image = composited
                overlaid = self.overlay.apply(
                    image,
                    text=overlay_text,
                    page_format=page_format,
                    text_image_relationship=text_rel,
                    element_count=count,
                    arrangement=structure.get("arrangement"),
                )
                pages.append(overlaid)
                continue

            # Standard single-image path
            prompt = self.prompt_builder.build(
                content=content,
                theme=theme,
                page_format=page_format,
                element_count=structure.get("element_count"),
                arrangement=structure.get("arrangement"),
                text_image_relationship=structure.get("text_image_relationship"),
            )

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

            structure = self.page_structure or {}
            text_rel = structure.get("text_image_relationship")
            overlaid = self.overlay.apply(
                image,
                text=content,
                page_format=page_format,
                text_image_relationship=text_rel,
                element_count=structure.get("element_count"),
                arrangement=structure.get("arrangement"),
            )
            pages.append(overlaid)

        # Verify every finished page against the example before compiling.
        # Opting out skips both the guard calls (avoiding the extra vision
        # API call per page) and the `Consistency check:` report.
        if self.verify_consistency:
            self.consistency_results = [self._verify_page(page) for page in pages]
            self._report_consistency()
        else:
            self.consistency_results = []
        return self.compiler.compile(pages, output_path_obj)

    def _verify_page(self, page: Image.Image) -> dict:
        """Run the consistency guard against one generated page.

        A guard failure (an unexpected extractor/classifier error) is recorded
        as an inconsistency instead of aborting generation: verification is a
        quality signal, never a hard requirement for producing the PDF.
        """
        try:
            return self.consistency_guard.verify(page)
        except Exception as exc:  # noqa: BLE001 - any check failure must not abort generation
            issue = f"Consistency check failed: {exc}"
            return {
                "consistent": False,
                "issues": [issue],
                "details": {"consistent": False, "issues": [issue]},
            }

    def _report_consistency(self) -> None:
        """Report consistency results: a one-line summary on stdout, plus a
        stderr warning for each issue on each inconsistent page."""
        results = self.consistency_results
        if not results:
            return
        matched = sum(1 for result in results if result.get("consistent"))
        total = len(results)
        print(f"Consistency check: {matched}/{total} page(s) match the example.")
        for index, result in enumerate(results, start=1):
            if result.get("consistent"):
                continue
            for issue in result.get("issues", []):
                print(f"Warning: page {index} is not consistent: {issue}", file=sys.stderr)
