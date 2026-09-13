"""CLI entrypoint for PageLoom."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pageloom import settings
from pageloom.content_generator import LLMContentGenerator
from pageloom.page_structure_classifier import detect_structure
from pageloom.pipeline import Pipeline
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
        description="PageLoom: generate children's book pages from an example image.",
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
    output_dir = settings.get_output_dir()
    default_output = str(Path(output_dir) / "output.pdf") if output_dir else "output.pdf"
    generate_parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Output PDF path.",
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

    if args.command != "generate":
        parser.print_help()
        return 1

    pages_value: int = args.pages
    if pages_value < 1:
        print("Error: --pages must be at least 1.", file=sys.stderr)
        return 1

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
            extra = llm_gen.generate(pages_value - built_in_limit, page_structure=page_structure)
            contents.extend(extra)
        except RuntimeError as exc:
            print(f"Error: LLM content generation failed: {exc}", file=sys.stderr)
            return 1
    else:
        contents = built_in_contents[:pages_value]

    pipeline = Pipeline(
        example_image=args.example,
        description=args.description,
        page_structure=page_structure,
    )
    result = pipeline.run(
        contents=contents,
        output_path=args.output,
        page_format=args.format,
        force=args.force,
    )
    print(f"Generated {len(contents)} page(s) -> {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
