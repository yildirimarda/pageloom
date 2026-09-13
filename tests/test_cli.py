"""Tests for the pageloom CLI entrypoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pageloom.cli import DEFAULT_CONTENTS, main


def test_cli_generate_pages_param(tmp_path: Path):
    """The --pages parameter controls how many pages are generated."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with patch("pageloom.cli.Pipeline") as mock_pipeline:
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "output.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "3",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
        assert result == 0

        mock_pipeline.assert_called_once_with(example_image=str(img_path))
        call_args = mock_instance.run.call_args
        assert call_args is not None
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        # Check that exactly 3 content items were passed
        assert len(contents) == 3
        assert contents == DEFAULT_CONTENTS[:3]


def test_cli_pages_less_than_one():
    result = main(["generate", "--pages", "0"])
    assert result == 1


def test_cli_pages_within_limit(tmp_path: Path):
    """Using the full available content list works."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)
    with patch("pageloom.cli.Pipeline") as mock_pipeline:
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "output.pdf"
        result = main(
            [
                "generate",
                "--pages",
                str(len(DEFAULT_CONTENTS)),
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
        assert result == 0
        contents = mock_instance.run.call_args.kwargs.get("contents")
        assert len(contents) == len(DEFAULT_CONTENTS)


def test_cli_pages_exceeds_content():
    """Requesting more pages than available raises a clear error."""
    result = main(["generate", "--pages", str(len(DEFAULT_CONTENTS) + 1)])
    assert result == 1


def test_cli_no_subcommand():
    result = main([])
    assert result == 1
