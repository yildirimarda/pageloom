"""CLI entrypoint for PageLoom."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pageloom import settings
from pageloom.content_generator import LLMContentGenerator
from pageloom.image_set import ImageSetRenderer, load_contents
from pageloom.page_structure_classifier import detect_structure
from pageloom.pdf_compiler import parse_page_size, write_individual_images, write_zip
from pageloom.pipeline import Pipeline
from pageloom.providers import IMAGE_PROVIDER_MAP, build_image_provider
from pageloom.providers.size_negotiation import generate_fitting_slot
from pageloom.quality_gate import QualityGateError
from pageloom.vision import PageVisionAnalyzer, VisionAnalysisError

# Default alphabet-style content list used when no explicit contents are given.
DEFAULT_CONTENTS = [
    "Apple",
    "Banana",
    "Cherry",
    "Dragonfruit",
    "Elderberry",
    "Fig",
    "Grape",
    "Honeydew",
    "Kiwi",
    "Lemon",
    "Mango",
    "Nectarine",
    "Orange",
    "Papaya",
    "Quince",
    "Raspberry",
    "Strawberry",
    "Tangerine",
    "Ugli fruit",
    "Vanilla bean",
    "Watermelon",
    "Xigua",
    "Yuzu",
    "Zucchini",
    "Apricot",
    "Blueberry",
    "Coconut",
    "Durian",
    "Eggplant",
    "Fennel",
    "Ginger",
    "Hazelnut",
    "Indigo",
    "Jackfruit",
    "Kale",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pageloom",
        description="PageLoom: generate pages matching an example image.",
    )
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate N pages from an example page.",
    )
    generate_parser.add_argument(
        "--pages",
        type=int,
        required=True,
        help="Number of pages to generate.",
    )
    generate_parser.add_argument(
        "--example",
        type=str,
        default="example.png",
        help="Path to the example page image.",
    )
    generate_parser.add_argument(
        "--description",
        type=str,
        default="",
        help=(
            "Optional free-text description of the example page (e.g. "
            "'object name pair' or 'alphabet letter word'). Passed to the "
            "page-structure classifier and pipeline to improve structural "
            "routing accuracy beyond pure image heuristics. Providing this "
            "skips automatic vision analysis (see --no-vision) since it's "
            "treated as an explicit override."
        ),
    )
    generate_parser.add_argument(
        "--no-vision",
        action="store_true",
        default=False,
        help=(
            "Disable automatic vision-based page-structure analysis. By default, "
            "when no --description is given, PageLoom sends the example image to "
            "a vision-capable LLM (OpenRouter's free tier by default) to "
            "automatically detect the page type -- numbers, letters, word-image "
            "matching, or anything else -- without you having to describe it. "
            "Pass this flag to fall back to the built-in image-size heuristic instead."
        ),
    )
    generate_parser.add_argument(
        "--vision-provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "openrouter"],
        help="Provider used for automatic vision analysis (default: openrouter, free tier).",
    )
    generate_parser.add_argument(
        "--no-consistency",
        action="store_true",
        default=False,
        help=(
            "Skip the automatic per-page consistency check. By default every "
            "generated page is verified against the example, which with vision "
            "enabled re-classifies each page (one extra vision API call per "
            "page on a first run) and prints a `Consistency check: N/M` "
            "summary. Pass this flag to skip verification and its report "
            "(mirrors --no-vision: same cost/reliability concern on OpenRouter's "
            "rate-capped free tier)."
        ),
    )
    output_dir = settings.get_output_dir()
    default_output = str(Path(output_dir) / "output.pdf") if output_dir else "output.pdf"
    generate_parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Output path (PDF by default; the stem is reused for numbered images/zip).",
    )
    generate_parser.add_argument(
        "--output-format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "jpeg", "zip"],
        help=(
            "Output format: 'pdf' compiles pages into a single PDF; 'png'/'jpeg' "
            "write individual numbered images next to --output; 'zip' bundles the "
            "numbered images into a zip archive at --output."
        ),
    )
    generate_parser.add_argument(
        "--format",
        type=str,
        default="standard",
        choices=["standard", "object_name", "themed_scene"],
        help="Page format.",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate raw illustrations even if cached.",
    )
    generate_parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Use optional LLM-based content generator when pages exceed the built-in list.",
    )
    generate_parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "openrouter"],
        help="LLM provider to use for content generation (default: openai).",
    )
    generate_parser.add_argument(
        "--image-provider",
        type=str,
        default=None,
        choices=list(IMAGE_PROVIDER_MAP),
        help="Image-generation provider (default: local, from PAGELOOM_IMAGE_PROVIDER).",
    )
    generate_parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Image model id (default: provider default, from PAGELOOM_IMAGE_MODEL).",
    )
    generate_parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=(
            "Maximum estimated total cost in USD for the run. Generation stops "
            "before this budget is exceeded (cached images are free)."
        ),
    )
    generate_parser.add_argument(
        "--estimate",
        action="store_true",
        default=False,
        help=(
            "Print the expected cost for the run without calling any image "
            "provider, then exit. Uses the configurable price table "
            "(PAGELOOM_IMAGE_PRICE_TABLE) and the detected page structure."
        ),
    )
    generate_parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help=(
            "Low-quality quick mode: generates with only a few inference steps "
            "(e.g. 6) for fast, rough previews. Clearly low quality -- never "
            "used for verification or consistency checks. Real runs use the "
            "production defaults (25-30 steps, DPM++ multistep scheduler, "
            "float16 on CUDA/MPS, guidance ~7)."
        ),
    )
    generate_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["fast", "quality"],
        help=(
            "Generation profile: 'fast' uses fewer inference steps for quick "
            "previews (6 steps, guidance ~5); 'quality' uses the production "
            "defaults (25-30 steps, guidance ~7). Defaults to 'quality' "
            "(overridable with PAGELOOM_PROFILE)."
        ),
    )
    generate_parser.add_argument(
        "--page-size",
        type=parse_page_size,
        default=None,
        help=(
            "Page size for the output PDF. Presets: square, a4, letter. "
            "Custom: WIDTHxHEIGHT in points (e.g. 612x792). Defaults to the "
            "example image's own size/aspect ratio."
        ),
    )

    check_vision_parser = subparsers.add_parser(
        "check-vision",
        help="Smoke-test the vision provider/model (one-shot validation).",
    )
    check_vision_parser.add_argument(
        "--vision-provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "openrouter"],
        help="Provider to smoke-test (default: openrouter).",
    )
    check_vision_parser.add_argument(
        "--example",
        type=str,
        default="",
        help="Optional example image path to test with; if omitted, a dummy image is used.",
    )

    check_image_parser = subparsers.add_parser(
        "check-image",
        help="Smoke-test the image provider (generate one small image).",
    )
    check_image_parser.add_argument(
        "--image-provider",
        type=str,
        default=None,
        choices=list(IMAGE_PROVIDER_MAP),
        help="Image provider to smoke-test (default: local, from PAGELOOM_IMAGE_PROVIDER).",
    )
    check_image_parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Image model id (default: provider default, from PAGELOOM_IMAGE_MODEL).",
    )

    images_parser = subparsers.add_parser(
        "images",
        help="Generate one image per prompt, with no page/letter/text-overlay logic.",
    )
    images_parser.add_argument(
        "prompts",
        nargs="*",
        help="Image prompts to render (one image per prompt).",
    )
    images_parser.add_argument(
        "--content-file",
        type=str,
        default=None,
        help=(
            "Path to a txt/csv/json file with one prompt per line/item "
            "(JSON: a list of strings, or objects with a ``prompt`` key). "
            "Prompts given as arguments are merged ahead of the file's items."
        ),
    )
    images_parser.add_argument(
        "--style-reference",
        type=str,
        default=None,
        help=(
            "Optional style reference image to condition generation on. "
            "Providers that support reference images use it to match its style."
        ),
    )
    images_parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output path (numbered images or a zip archive next to it).",
    )
    images_parser.add_argument(
        "--output-format",
        type=str,
        default="png",
        choices=["png", "jpeg", "zip"],
        help="Output format: 'png'/'jpeg' write numbered images; 'zip' bundles them.",
    )
    images_parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Generation width in pixels (rounded up to a multiple of 8).",
    )
    images_parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Generation height in pixels (rounded up to a multiple of 8).",
    )
    images_parser.add_argument(
        "--image-provider",
        type=str,
        default=None,
        choices=list(IMAGE_PROVIDER_MAP),
        help="Image-generation provider (default: local, from PAGELOOM_IMAGE_PROVIDER).",
    )
    images_parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Image model id (default: provider default, from PAGELOOM_IMAGE_MODEL).",
    )
    images_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate images even if a cached copy exists.",
    )
    images_parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=(
            "Maximum estimated total cost in USD for the run. Generation stops "
            "before this budget is exceeded (cached images are free)."
        ),
    )
    images_parser.add_argument(
        "--estimate",
        action="store_true",
        default=False,
        help=("Print the expected cost for the run without calling any image provider, then exit."),
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "check-vision":
        try:
            vision_analyzer = PageVisionAnalyzer(provider=args.vision_provider)
        except (ValueError, VisionAnalysisError) as exc:
            print(f"FAIL: Vision provider setup failed: {exc}", file=sys.stderr)
            return 1

        print(f"Provider: {vision_analyzer.provider_name}")
        print(f"Model:    {vision_analyzer.provider.model}")
        if getattr(vision_analyzer.provider, "endpoint", None):
            print(f"Endpoint: {vision_analyzer.provider.endpoint}")

        # One-shot smoke test: use the provided example or a temporary dummy PNG.
        import tempfile

        from PIL import Image

        if args.example:
            test_image_path = args.example
        else:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                test_image_path = f.name
            Image.new("RGB", (256, 256), color="red").save(test_image_path)

        try:
            result = vision_analyzer.analyze(source=test_image_path, description="")
            print(f"Smoke test: PASS (vision analysis returned {len(result)} fields)")
            print(f"Detected pattern: {result.get('content_pattern', 'N/A')}")
            print(f"Element count:   {result.get('element_count', 'N/A')}")
            return 0
        except VisionAnalysisError as exc:
            print(f"FAIL: Vision smoke test failed: {exc}", file=sys.stderr)
            return 1
        except (ValueError, OSError) as exc:
            print(f"FAIL: Unexpected error during smoke test: {exc}", file=sys.stderr)
            return 1

    if args.command == "check-image":
        try:
            generator = build_image_provider(provider=args.image_provider, model=args.image_model)
        except (ValueError, RuntimeError) as exc:
            print(f"FAIL: Image provider setup failed: {exc}", file=sys.stderr)
            return 1

        print(f"Provider: {generator.provider_name}")
        print(f"Model:    {generator.model_id}")

        # One-shot smoke test: generate one small image through size
        # negotiation, exactly like a real page generation.
        from PIL import Image

        try:
            image = generate_fitting_slot(
                generator,
                prompt="A simple solid-color square for a provider smoke test.",
                width=256,
                height=256,
            )
            if not isinstance(image, Image.Image):
                raise TypeError("Provider did not return a PIL image.")
            print("Smoke test: PASS (one small image generated successfully)")
            return 0
        except Exception as exc:  # noqa: BLE001 - surface the exact failure
            print(f"FAIL: Image smoke test failed: {exc}", file=sys.stderr)
            return 1

    if args.command == "images":
        return _run_images(args)

    if args.command != "generate":
        parser.print_help()
        return 1

    pages_value: int = args.pages
    if pages_value < 1:
        print("Error: --pages must be at least 1.", file=sys.stderr)
        return 1

    # Apply --profile for this run (overrides PAGELOOM_PROFILE), restoring
    # afterwards so the setting does not leak beyond this process invocation.
    _prev_profile = os.environ.get("PAGELOOM_PROFILE")
    if args.profile is not None:
        os.environ["PAGELOOM_PROFILE"] = args.profile
    try:
        # Automatic vision analysis is the default way to detect page structure: it
        # looks at the example image itself, so it works for ANY page type (numbers,
        # letters, word-image matching, etc.) without the user writing a description.
        # An explicit --description is treated as an override that skips the (network)
        # vision call; --no-vision opts out entirely in favor of the old heuristic.
        vision_analyzer = None
        if not args.no_vision and not args.description:
            try:
                vision_analyzer = PageVisionAnalyzer(provider=args.vision_provider)
            except (ValueError, VisionAnalysisError) as exc:
                print(
                    f"Warning: automatic vision analysis unavailable ({exc}); "
                    f"falling back to image-size heuristics. Pass --description to "
                    f"improve accuracy, or set up a free OpenRouter key (see README).",
                    file=sys.stderr,
                )

        # Detect page structure from the example image (plus optional description)
        # to route LLM content correctly.
        page_structure = None
        try:
            page_structure = detect_structure(
                source=args.example, description=args.description, vision_analyzer=vision_analyzer
            )
        except (ValueError, OSError, FileNotFoundError):
            # If classification fails, fall back to generic single-letter-word structure
            page_structure = {
                "content_pattern": "single_letter_word_illustration",
                "element_count": 1,
                "arrangement": "centered",
                "text_image_relationship": "label_per_image",
            }

        # --estimate: print the expected cost for the run without calling any image
        # provider, then exit. The price table (PAGELOOM_IMAGE_PRICE_TABLE) and the
        # detected page structure determine the per-image cost.
        if args.estimate:
            estimate_pipeline = Pipeline(
                example_image=args.example,
                description=args.description,
                provider=args.image_provider,
                model=args.image_model,
                page_structure=page_structure,
                verify_consistency=False,
            )
            estimate = estimate_pipeline.estimated_total_cost_usd(pages_value)
            print(f"Estimated cost: ${estimate:.2f} for {pages_value} page(s)")
            return 0

        # Determine available content based on detected page structure.
        # Only alphabet-style pages have a fixed built-in content list.
        content_pattern = page_structure.get("content_pattern", "single_letter_word_illustration")
        if content_pattern == "single_letter_word_illustration":
            built_in_contents = DEFAULT_CONTENTS
            built_in_limit = len(DEFAULT_CONTENTS)
        else:
            # Non-alphabet structures (object_name_pair, multi_item_scene,
            # etc.) have no fixed built-in content list.
            built_in_contents = []
            built_in_limit = 0

        # The LLM generator should trigger whenever N exceeds whatever content
        # items are explicitly available for the detected structure -- not a
        # hardcoded 26-letter boundary that only applies to the legacy alphabet
        # list.
        use_llm = args.use_llm
        if pages_value > built_in_limit:
            if not use_llm:
                print(
                    f"Error: --pages ({pages_value}) exceeds available content "
                    f"({built_in_limit} items). Use --use-llm to generate extra content.",
                    file=sys.stderr,
                )
                return 1
            contents = built_in_contents[:built_in_limit]
            try:
                llm_gen = LLMContentGenerator(provider=args.llm_provider)
                extra = llm_gen.generate(
                    pages_value - built_in_limit, page_structure=page_structure
                )
                contents.extend(extra)
            except RuntimeError as exc:
                print(f"Error: LLM content generation failed: {exc}", file=sys.stderr)
                return 1
        else:
            contents = built_in_contents[:pages_value]

        pipeline = Pipeline(
            example_image=args.example,
            description=args.description,
            provider=args.image_provider,
            model=args.image_model,
            page_structure=page_structure,
            # Same analyzer used to classify the example: Pipeline forwards it to
            # ConsistencyGuard so generated pages are re-classified with the same
            # method (vision or heuristic) during the automatic consistency check.
            vision_analyzer=vision_analyzer,
            # --no-consistency opts out of that check entirely (no guard runs and
            # no `Consistency check:` report). --quick is low quality and is never
            # used for verification, so it also skips the consistency check.
            verify_consistency=not args.no_consistency and not args.quick,
            # --max-cost caps total estimated spend; the run stops before exceeding it.
            max_cost_usd=args.max_cost,
            page_size=args.page_size,
        )
        if args.quick:
            # Quick mode is low quality: drop to a rough-preview step count. The
            # production defaults (25-30 steps) are used for real runs.
            pipeline.generator.num_inference_steps = 6

        try:
            result = pipeline.run(
                contents=contents,
                output_path=args.output,
                page_format=args.format,
                force=args.force,
                output_format=args.output_format,
            )
        except QualityGateError as exc:
            print(
                f"Error: a generated frame failed the quality gate and was rejected "
                f"({', '.join(exc.reasons)}). No bad image was written to the PDF. "
                f"Try a different prompt/example or raise PAGELOOM_QUALITY_GATE_MAX_RETRIES.",
                file=sys.stderr,
            )
            return 1
        print(f"Generated {len(contents)} page(s) -> {result}")
        return 0
    finally:
        if _prev_profile is None:
            os.environ.pop("PAGELOOM_PROFILE", None)
        else:
            os.environ["PAGELOOM_PROFILE"] = _prev_profile


def _run_images(args) -> int:
    """Handle the ``pageloom images`` subcommand.

    Generates one image per prompt (from arguments or a content file), optionally
    conditioned on a style reference, with no page, letter, or text-overlay
    logic. Writes numbered images or a zip archive to --output.
    """
    try:
        contents = load_contents(args.prompts, args.content_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not contents:
        print(
            "Error: no prompts given. Pass prompts as arguments or use --content-file.",
            file=sys.stderr,
        )
        return 1

    renderer = ImageSetRenderer(
        contents=contents,
        provider_name=args.image_provider,
        model=args.image_model,
        style_reference=args.style_reference,
        width=args.width,
        height=args.height,
        force=args.force,
        max_cost_usd=args.max_cost,
    )
    if args.estimate:
        print(
            f"Estimated cost: ${renderer.estimated_total_cost_usd():.2f} for {len(contents)} image(s)"
        )
        return 0

    try:
        images = renderer.render()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: image generation failed: {exc}", file=sys.stderr)
        return 1
    if not images:
        print("Warning: no images were generated (budget may have been reached).", file=sys.stderr)
        return 0

    output_path = Path(args.output)
    output_dir = settings.get_output_dir()
    if output_dir is not None and not output_path.is_absolute():
        output_path = Path(output_dir) / output_path

    try:
        if args.output_format == "zip":
            result = write_zip(images, output_path)
        else:
            result = write_individual_images(images, output_path, args.output_format)
    except (OSError, ValueError) as exc:
        print(f"Error: could not write output: {exc}", file=sys.stderr)
        return 1
    print(f"Generated {len(images)} image(s) -> {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
