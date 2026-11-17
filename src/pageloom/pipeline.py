"""Main pipeline orchestration that infers layout and style directly from an example page."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

from pageloom import settings
from pageloom.consistency import ConsistencyGuard
from pageloom.overlay import TextOverlay
from pageloom.page_structure_classifier import detect_structure
from pageloom.pdf_compiler import PDFCompiler
from pageloom.providers import ImageProvider, ImageProviderCapabilities, build_image_provider
from pageloom.providers.size_negotiation import generate_fitting_slot, negotiate_size
from pageloom.quality_gate import QualityGateError, quality_gate_reasons
from pageloom.style import StylePromptBuilder
from pageloom.style_extractor import VisualStyleExtractor
from pageloom.vision import PageVisionAnalyzer


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
        generator: ImageProvider | None = None,
        provider: str | None = None,
        model: str | None = None,
        compiler: PDFCompiler | None = None,
        page_structure: dict | None = None,
        vision_analyzer: object | None = None,
        verify_consistency: bool = True,
        max_cost_usd: float | None = None,
        quality_gate_max_retries: int | None = None,
        page_size: tuple[float, float] | None = None,
    ) -> None:
        self.extractor = VisualStyleExtractor(example_image, description=description)
        self.style = self.extractor.extract()
        # Detect page structure for full structural routing through the pipeline.
        # A pre-computed `page_structure` (e.g. one the CLI already obtained via
        # vision analysis) is used as-is to avoid a second, redundant analysis
        # call; otherwise detection runs now, preferring `vision_analyzer` when
        # given and falling back to the heuristic classifier automatically.
        self.page_structure = page_structure or detect_structure(
            source=example_image, description=description, vision_analyzer=vision_analyzer
        )
        # Merge the vision model's real style description (medium, line weight,
        # palette in colour names, background, lighting) into the style profile
        # so StylePromptBuilder builds the diffusion prompt from the example's
        # actual style instead of the layout label. Heuristic-fallback pages
        # have no ``style`` key, so merge defensively; the builder keeps the
        # layout-label fallback when no vision style is available.
        vision_style = self.page_structure.get("style") or {}
        if isinstance(vision_style, dict) and vision_style:
            self.style["style"] = vision_style
        self.prompt_builder = StylePromptBuilder(
            example_style=self.style,
        )
        # Resolve the image provider from the explicit `generator`, then the
        # CLI/provider/model arguments, then the settings default (``local``).
        self.generator = generator or build_image_provider(provider=provider, model=model)
        self.overlay = TextOverlay(style_profile=self.style)
        # Page size for the compiled PDF: an explicit page_size wins, otherwise
        # default to the example image's own size (its aspect ratio).
        if page_size is None:
            example_width, example_height = self.extractor.image.size
            page_size = (float(example_width), float(example_height))
        self.page_size = page_size
        self.compiler = compiler or PDFCompiler(page_size=self.page_size)
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
        # Cost awareness: per-image estimated cost from the configurable price
        # table, summed across the run. ``max_cost_usd`` caps total spend.
        self.max_cost_usd = max_cost_usd
        self.total_cost_usd = 0.0
        self.budget_stopped = False
        # Quality gate: cheap-heuristic retries with a fresh seed before a
        # frame is rejected. ``None`` falls back to the setting.
        self.quality_gate_max_retries = (
            settings.get_quality_gate_max_retries()
            if quality_gate_max_retries is None
            else quality_gate_max_retries
        )
        # The first accepted generated page becomes a style reference for
        # subsequent pages when the provider supports reference images.
        self._first_generated_page: Image.Image | None = None

    def _image_cost_usd(self, width: int, height: int) -> float:
        """Estimated USD cost for one image at the provider-negotiated size."""
        capabilities = getattr(self.generator, "capabilities", None)
        gen_width, gen_height = negotiate_size(int(width), int(height), capabilities)
        return self.generator.estimated_cost_usd(gen_width, gen_height)

    def estimated_total_cost_usd(self, num_pages: int) -> float:
        """Expected USD cost for generating ``num_pages`` pages, without calling
        the provider.

        Reads the per-image price from the configurable price table and scales
        by the number of images each page requires (``multiple_labels`` pages
        generate one sub-image per element). Cached images are not discounted:
        this is a fresh-run estimate.
        """
        structure = self.page_structure or {}
        text_rel = structure.get("text_image_relationship")
        images_per_page = 1
        if text_rel == "multiple_labels":
            images_per_page = max(1, int(structure.get("element_count", 1)))
        width = round_to_multiple(
            self.style.get("composition", {}).get("page_dimensions", {}).get("width", 612)
        )
        height = round_to_multiple(
            self.style.get("composition", {}).get("page_dimensions", {}).get("height", 612)
        )
        per_image = self.generator.estimated_cost_usd(width, height)
        return num_pages * images_per_page * per_image

    def _generate_accepted_image(
        self,
        generator,
        prompt: str,
        width: int,
        height: int,
        content: str,
        references: list[str | Image.Image] | None = None,
    ) -> Image.Image:
        """Generate one image through the quality gate.

        The image is rejected (and regenerated with a fresh seed) when any cheap
        heuristic flags it, or when the optional vision subject check denies the
        requested ``content``. Retries happen up to ``quality_gate_max_retries``
        times; a frame that never passes raises :class:`QualityGateError` so it
        is never written into the PDF.
        """
        max_retries = int(self.quality_gate_max_retries)
        last_image: Image.Image | None = None
        last_reasons: list[str] = []
        for attempt in range(max_retries + 1):
            attempt_seed = None  # each attempt is a fresh random draw
            image = generate_fitting_slot(
                generator,
                prompt=prompt,
                width=width,
                height=height,
                references=references,
                seed=attempt_seed,
            )
            reasons = quality_gate_reasons(image)
            if not reasons:
                if self._subject_present(image, content):
                    return image
                # The cheap heuristics passed but the vision model denies the
                # requested subject: treat that as a failure reason and retry.
                reasons.append("subject not present")
            last_image = image
            last_reasons = reasons
            if attempt < max_retries:
                print(
                    f"Warning: image failed quality gate ({', '.join(reasons)}); "
                    f"retrying ({attempt + 1}/{max_retries}) with a new seed.",
                    file=sys.stderr,
                )
                continue
        raise QualityGateError(last_reasons, image=last_image)

    def _subject_present(self, image: Image.Image, content: str) -> bool:
        """Optionally ask the vision model whether ``content``'s subject is visible.

        Returns True (no check applied) when no vision analyser is configured,
        so offline runs are unaffected.
        """
        if self.vision_analyzer is None:
            return True
        provider = getattr(self.vision_analyzer, "provider", None)
        if provider is None or not hasattr(provider, "generate_vision_content"):
            # A stub analyser (e.g. one that only implements `analyze()`) cannot
            # answer subject questions, so skip the optional subject check.
            return True
        data_url = PageVisionAnalyzer._encode_image(image)
        prompt = f"Is the subject '{content}' present in this image? Answer yes or no."
        raw = provider.generate_vision_content(
            prompt=prompt,
            image_data_url=data_url,
            max_tokens=8,
            temperature=0.2,
        )
        return raw.strip().lower().startswith("y")

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
        # Warn before a slow CPU run: Docker on macOS cannot use the MPS
        # (Apple Silicon) backend, so local Stable Diffusion inside Docker
        # falls back to CPU and can take minutes per page. Suggest plain
        # `uv run` (MPS-capable on Apple Silicon) or an online provider.
        device = getattr(self.generator, "device", None)
        if device == "cpu":
            structure = self.page_structure or {}
            images_per_page = 1
            if structure.get("text_image_relationship") == "multiple_labels":
                images_per_page = max(1, int(structure.get("element_count", 1)))
            steps = getattr(self.generator, "num_inference_steps", None)
            if steps is None:
                steps = settings.get_num_inference_steps()
            estimated = settings.estimated_cpu_run_seconds(len(contents), steps, images_per_page)
            threshold = settings.get_cpu_slow_warn_threshold_seconds()
            if estimated > threshold:
                print(
                    "Warning: this run is estimated to take about "
                    f"{estimated:.0f}s on CPU ({len(contents)} page(s) x {steps} "
                    "step(s)). Docker on macOS cannot use the MPS (Apple "
                    "Silicon) backend, so local Stable Diffusion inside Docker "
                    "runs slowly on CPU. For faster generation, run locally "
                    "with plain `uv run` (which can use MPS) or switch to an "
                    "online image provider "
                    "(--image-provider openai/gemini/openrouter/flux). "
                    "Set PAGELOOM_CPU_SLOW_WARN_SECONDS=0 to silence this "
                    "warning.",
                    file=sys.stderr,
                )
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

                    # Generate sub-image at a provider-native size, then fit it
                    # back to the sub-slot (skip individual sub-caching). Stop before
                    # the provider call if this sub-image would exceed the budget.
                    sub_image_cost = self._image_cost_usd(
                        round_to_multiple(sub_w), round_to_multiple(sub_h)
                    )
                    if (
                        self.max_cost_usd is not None
                        and self.total_cost_usd + sub_image_cost > self.max_cost_usd
                    ):
                        self.budget_stopped = True
                        break
                    sub_image = self._generate_accepted_image(
                        self.generator,
                        prompt=sub_prompt,
                        width=round_to_multiple(sub_w),
                        height=round_to_multiple(sub_h),
                        content=label_content,
                    )
                    sub_images.append(sub_image)
                    self.total_cost_usd += sub_image_cost

                if self.budget_stopped:
                    break

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

            # Standard single-image path. When the provider accepts reference
            # images, the example page and the first accepted generated page
            # are sent as references with a style-matching instruction instead
            # of the text-only style description; otherwise the text path is
            # used (page 1 always starts text-only since no reference exists yet).
            capabilities = getattr(self.generator, "capabilities", None)
            supports_references = (
                isinstance(capabilities, ImageProviderCapabilities)
                and capabilities.max_reference_images > 0
            )
            references: list[str | Image.Image] | None = None
            if supports_references and self._first_generated_page is not None:
                references = [self.extractor.image, self._first_generated_page]

            prompt = self.prompt_builder.build(
                content=content,
                theme=theme,
                page_format=page_format,
                element_count=structure.get("element_count"),
                arrangement=structure.get("arrangement"),
                text_image_relationship=structure.get("text_image_relationship"),
                use_reference_style=bool(references),
            )

            cache_key = f"{content}:{theme}:{page_format}:{int(width)}:{int(height)}:{prompt}"
            cache_filename = hashlib.md5(cache_key.encode("utf-8")).hexdigest() + ".png"
            cache_path = cache_dir / cache_filename

            if not force and cache_path.exists():
                image = Image.open(str(cache_path))
            else:
                # Negotiate a provider-native size, generate, then crop/pad
                # back to the layout slot (size negotiation). Stop before the
                # provider call if this image would exceed the max-cost budget.
                image_cost = self._image_cost_usd(int(width), int(height))
                if (
                    self.max_cost_usd is not None
                    and self.total_cost_usd + image_cost > self.max_cost_usd
                ):
                    self.budget_stopped = True
                    break
                # Generate with the automatic quality gate: reject near-black,
                # near-uniform, tiled or noise-like frames, optionally confirm
                # the requested subject with the vision model, and retry with a
                # new seed up to the configured limit. A frame that never
                # passes is reported as a failure and never written to the PDF.
                image = self._generate_accepted_image(
                    self.generator,
                    prompt=prompt,
                    width=int(width),
                    height=int(height),
                    references=references,
                    content=content,
                )
                image.save(str(cache_path), "PNG")
                self.total_cost_usd += image_cost

            # Remember the first generated page so later pages can reference it
            # for style consistency (page 1 is always generated text-only since
            # no reference exists yet; from page 2 onward references are used).
            if self._first_generated_page is None:
                self._first_generated_page = image

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

        if self.budget_stopped and self.max_cost_usd is not None:
            print(
                f"Warning: max cost ${self.max_cost_usd:.2f} reached; stopped "
                f"generation before exceeding the budget "
                f"(${self.total_cost_usd:.2f} of ${self.max_cost_usd:.2f} used).",
                file=sys.stderr,
            )

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
