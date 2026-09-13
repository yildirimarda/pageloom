"""CLI entrypoint for PageLoom."""

from __future__ import annotations

import argparse
import sys

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
    generate_parser.add_argument(
        "--output",
        type=str,
        default="output.pdf",
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
    if pages_value > len(DEFAULT_CONTENTS):
        print(
            f"Error: --pages ({pages_value}) exceeds available content "
            f"({len(DEFAULT_CONTENTS)} items).",
            file=sys.stderr,
        )
        return 1

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
