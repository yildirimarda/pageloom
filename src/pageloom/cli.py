"""CLI entrypoint for PageLoom."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pageloom import settings
from pageloom.content_generator import LLMContentGenerator
from pageloom.pipeline import Pipeline

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

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command != "generate":
        parser.print_help()
        return 1

    pages_value: int = args.pages
    if pages_value < 1:
        print("Error: --pages must be at least 1.", file=sys.stderr)
        return 1

    # The built-in alphabet list covers the first 26 items; beyond that we can
    # optionally use the LLM-based generator.
    ALPHABET_LIMIT = 26
    use_llm = args.use_llm
    if pages_value > len(DEFAULT_CONTENTS):
        if not use_llm:
            print(
                f"Error: --pages ({pages_value}) exceeds available content "
                f"({len(DEFAULT_CONTENTS)} items). Use --use-llm to generate extra content.",
                file=sys.stderr,
            )
            return 1
        contents = DEFAULT_CONTENTS[: len(DEFAULT_CONTENTS)]
        try:
            llm_gen = LLMContentGenerator()
            extra = llm_gen.generate(pages_value - len(DEFAULT_CONTENTS))
            contents.extend(extra)
        except RuntimeError as exc:
            print(f"Error: LLM content generation failed: {exc}", file=sys.stderr)
            return 1
    elif pages_value > ALPHABET_LIMIT:
        contents = DEFAULT_CONTENTS[:pages_value]
        if use_llm:
            # When explicitly requested, extend with LLM-generated content past 26
            try:
                llm_gen = LLMContentGenerator()
                extra = llm_gen.generate(pages_value - ALPHABET_LIMIT)
                contents = DEFAULT_CONTENTS[:ALPHABET_LIMIT] + extra
            except RuntimeError as exc:
                print(f"Error: LLM content generation failed: {exc}", file=sys.stderr)
                return 1
        else:
            contents = DEFAULT_CONTENTS[:pages_value]
    else:
        contents = DEFAULT_CONTENTS[:pages_value]

    pipeline = Pipeline(example_image=args.example)
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
