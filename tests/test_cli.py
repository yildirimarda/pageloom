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


def test_cli_force_flag_passed_to_pipeline(tmp_path: Path):
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
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--force",
            ]
        )
        assert result == 0
        call_kwargs = mock_instance.run.call_args.kwargs
        assert call_kwargs.get("force") is True


def test_cli_use_llm_extends_past_alphabet_limit(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Dragon, Unicorn, Whale"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch("pageloom.cli.Pipeline") as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance
            mock_instance.run.return_value = tmp_path / "output.pdf"

            result = main(
                [
                    "generate",
                    "--pages",
                    "28",
                    "--example",
                    str(img_path),
                    "--output",
                    str(tmp_path / "out.pdf"),
                    "--use-llm",
                ]
            )
            assert result == 0
            contents = mock_instance.run.call_args.kwargs.get("contents")
            assert contents is not None
            assert len(contents) == 28
            assert contents[:26] == DEFAULT_CONTENTS[:26]
            assert contents[26:] == ["Dragon", "Unicorn"]
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_cli_use_llm_exceeds_full_content_list(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Extra1, Extra2"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with patch("pageloom.cli.Pipeline") as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance
            mock_instance.run.return_value = tmp_path / "output.pdf"

            result = main(
                [
                    "generate",
                    "--pages",
                    "52",
                    "--example",
                    str(img_path),
                    "--output",
                    str(tmp_path / "out.pdf"),
                    "--use-llm",
                ]
            )
            assert result == 0
            contents = mock_instance.run.call_args.kwargs.get("contents")
            assert len(contents) == 52
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_cli_use_llm_failure_returns_error(tmp_path: Path):
    result = main(["generate", "--pages", "28", "--use-llm"])
    assert result == 1


def test_cli_no_subcommand():
    result = main([])
    assert result == 1
