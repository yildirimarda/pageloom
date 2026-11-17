"""Tests for the pageloom CLI entrypoint."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pageloom.cli import DEFAULT_CONTENTS, main
from pageloom.consistency import ConsistencyGuard
from pageloom.vision import VisionAnalysisError


def _passing_image(width: int, height: int) -> Image.Image:
    """A non-uniform illustration that passes the automatic quality gate.

    Solid-colour mock images are now legitimately rejected (they are
    near-uniform), so a test that represents a successful generation must return
    a non-uniform image. A horizontal gradient has varying block means (not
    tiled), low adjacent-pixel difference (not noise-like), and is neither
    black nor uniform.
    """
    img = Image.new("L", (width, height))
    img.putdata([int(255 * x / max(1, width - 1)) for y in range(height) for x in range(width)])
    return img.convert("RGB")


# Fixed vision result used to keep the generate-path overflow tests hermetic
# (PLAN.md item 86): without it those tests make real openrouter/free calls
# whose non-deterministic outcome (sometimes a valid, non-alphabet JSON) would
# break their content-routing assertions. The structure is the alphabet page
# type so the full built-in content list (DEFAULT_CONTENTS) stays available,
# preserving exactly what each test asserts.
_DEFAULT_VISION_STRUCTURE = {
    "content_pattern": "single_letter_word_illustration",
    "element_count": 1,
    "arrangement": "centered",
    "text_image_relationship": "label_per_image",
    "content_type_description": "a letter and word page",
    "example_content_items": ["Apple"],
}


def test_cli_generate_pages_param(tmp_path: Path):
    """The --pages parameter controls how many pages are generated."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    detected_structure = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure),
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
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

        mock_pipeline.assert_called_once_with(
            example_image=str(img_path),
            description="",
            page_structure=detected_structure,
            provider=None,
            model=None,
            vision_analyzer=mock_vision_cls.return_value,
            verify_consistency=True,
            max_cost_usd=None,
            page_size=None,
        )
        call_args = mock_instance.run.call_args
        assert call_args is not None
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        # Check that exactly 3 content items were passed
        assert len(contents) == 3
        assert contents == DEFAULT_CONTENTS[:3]


def test_cli_generate_content_file_txt(tmp_path: Path):
    """`generate --content-file` supplies items from a plain-text file without an LLM."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    content_file = tmp_path / "items.txt"
    content_file.write_text("Alpha\nBeta\nGamma\n", encoding="utf-8")

    detected_structure = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure),
        patch("pageloom.cli.PageVisionAnalyzer"),
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--content-file",
                str(content_file),
            ]
        )
        assert result == 0
        contents = mock_instance.run.call_args.kwargs.get("contents")
        assert contents == ["Alpha", "Beta"]


def test_cli_generate_content_file_json(tmp_path: Path):
    """`generate --content-file` supports JSON lists of strings without an LLM."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    content_file = tmp_path / "items.json"
    content_file.write_text('["Alpha", "Beta", "Gamma"]', encoding="utf-8")

    detected_structure = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure),
        patch("pageloom.cli.PageVisionAnalyzer"),
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--content-file",
                str(content_file),
            ]
        )
        assert result == 0
        contents = mock_instance.run.call_args.kwargs.get("contents")
        assert contents == ["Alpha", "Beta"]


def test_cli_generate_content_file_json_objects(tmp_path: Path):
    """`generate --content-file` supports JSON objects with a ``prompt`` key."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    content_file = tmp_path / "items.json"
    content_file.write_text('[{"prompt": "Alpha"}, {"prompt": "Beta"}]', encoding="utf-8")

    detected_structure = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure),
        patch("pageloom.cli.PageVisionAnalyzer"),
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--content-file",
                str(content_file),
            ]
        )
        assert result == 0
        contents = mock_instance.run.call_args.kwargs.get("contents")
        assert contents == ["Alpha", "Beta"]


def test_cli_generate_content_file_invalid_json_names_line(tmp_path: Path, capsys):
    """Invalid JSON in a content file names the offending line and column."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    content_file = tmp_path / "items.json"
    content_file.write_text('[\n  "Alpha",\n  not valid\n]', encoding="utf-8")

    detected_structure = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure),
        patch("pageloom.cli.PageVisionAnalyzer"),
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--content-file",
                str(content_file),
            ]
        )
        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.err
        assert "line 3" in captured.err


def test_cli_generate_content_file_bad_entry_names_line(tmp_path: Path, capsys):
    """A non-string, non-prompt JSON entry names its line."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    content_file = tmp_path / "items.json"
    content_file.write_text('["Alpha", 42, "Beta"]', encoding="utf-8")

    detected_structure = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure),
        patch("pageloom.cli.PageVisionAnalyzer"),
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--content-file",
                str(content_file),
            ]
        )
        assert result == 1
        captured = capsys.readouterr()
        assert "line 2" in captured.err
        assert "Unsupported JSON entry" in captured.err


def test_cli_generate_content_file_missing(tmp_path: Path, capsys):
    """A missing content file exits 1 with a clear error naming the file."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    detected_structure = {
        "content_pattern": "single_letter_word_illustration",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure),
        patch("pageloom.cli.PageVisionAnalyzer"),
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        missing = tmp_path / "does-not-exist.txt"
        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--content-file",
                str(missing),
            ]
        )
        assert result == 1
        captured = capsys.readouterr()
        assert "Content file not found" in captured.err


def test_cli_quick_flag_skips_verification_and_uses_low_steps(tmp_path: Path):
    """`--quick` is clearly low quality: it skips the consistency check and
    drops the generator to a rough-preview step count. Never used for
    verification; real runs keep the production defaults (25-30 steps)."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    with (
        patch("pageloom.cli.detect_structure", return_value=_DEFAULT_VISION_STRUCTURE),
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "3",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--quick",
            ]
        )
        assert result == 0

        # --quick must never be used for verification.
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["verify_consistency"] is False

        # --quick drops to a low step count (rough preview), not the production default.
        assert mock_instance.generator.num_inference_steps == 6


def test_cli_profile_fast_and_quality_are_accepted(tmp_path: Path):
    """`--profile fast` and `--profile quality` are accepted on the generate path."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    with (
        patch("pageloom.cli.detect_structure", return_value=_DEFAULT_VISION_STRUCTURE),
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        for profile in ("fast", "quality"):
            result = main(
                [
                    "generate",
                    "--pages",
                    "2",
                    "--example",
                    str(img_path),
                    "--output",
                    str(tmp_path / "out.pdf"),
                    "--profile",
                    profile,
                ]
            )
            assert result == 0

        # --profile is applied (restored afterwards so it does not leak to
        # other tests): PAGELOOM_PROFILE must not remain set after main() returns.
        assert "PAGELOOM_PROFILE" not in os.environ


def test_cli_profile_invalid_is_rejected(tmp_path: Path):
    """An invalid `--profile` value is rejected by argparse."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--profile",
                "bogus",
            ]
        )
    assert exc_info.value.code == 2


def test_cli_page_size_presets_and_custom_passed_to_pipeline(tmp_path: Path):
    """`--page-size` accepts presets (square, a4, letter) and WIDTHxHEIGHT and
    forwards the parsed size to Pipeline as page_size."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    with (
        patch("pageloom.cli.detect_structure", return_value=_DEFAULT_VISION_STRUCTURE),
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        for page_size in ("square", "a4", "letter", "612x792"):
            result = main(
                [
                    "generate",
                    "--pages",
                    "1",
                    "--example",
                    str(img_path),
                    "--output",
                    str(tmp_path / "out.pdf"),
                    "--page-size",
                    page_size,
                ]
            )
            assert result == 0
            kwargs = mock_pipeline.call_args.kwargs
            assert kwargs["page_size"] is not None

        # Default (no --page-size) leaves page_size unset (None -> example's own size).
        main(
            [
                "generate",
                "--pages",
                "1",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
        assert mock_pipeline.call_args.kwargs["page_size"] is None


def test_cli_page_size_invalid_is_rejected(tmp_path: Path):
    """An invalid `--page-size` value is rejected by argparse."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--page-size",
                "bogus",
            ]
        )
    assert exc_info.value.code == 2


def test_cli_pages_within_limit(tmp_path: Path):
    """Using the full available content list works."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)
    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
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
    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
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


def test_cli_pages_within_full_built_in_list_never_calls_llm(tmp_path: Path):
    """Requesting up to the full built-in content list (35 items) should use the
    built-in content directly and must NOT invoke the LLM generator, even when
    --use-llm is passed -- the overflow trigger is the true content-availability
    boundary (len(DEFAULT_CONTENTS)), not the legacy 26-letter alphabet count."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
        patch("pageloom.cli.LLMContentGenerator") as mock_llm_gen,
    ):
        mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
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
        mock_llm_gen.assert_not_called()
        contents = mock_instance.run.call_args.kwargs.get("contents")
        assert contents == DEFAULT_CONTENTS[:28]


def test_cli_use_llm_extends_past_full_built_in_list(tmp_path: Path):
    """Once --pages exceeds the full built-in list (35 items), the LLM generator
    should be invoked for exactly the shortfall, on top of all 35 built-in items."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Dragon, Unicorn"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with (
            patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("pageloom.cli.Pipeline") as mock_pipeline,
        ):
            mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance
            mock_instance.run.return_value = tmp_path / "output.pdf"

            result = main(
                [
                    "generate",
                    "--pages",
                    str(len(DEFAULT_CONTENTS) + 2),
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
            assert len(contents) == len(DEFAULT_CONTENTS) + 2
            assert contents[: len(DEFAULT_CONTENTS)] == DEFAULT_CONTENTS
            assert contents[len(DEFAULT_CONTENTS) :] == ["Dragon", "Unicorn"]
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
    mock_response.choices = [
        MagicMock(message=MagicMock(content=", ".join([f"Extra{i}" for i in range(1, 27)])))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_module.OpenAI.return_value = mock_client

    import sys

    original_modules = dict(sys.modules)
    sys.modules["openai"] = mock_openai_module
    try:
        with (
            patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("pageloom.cli.Pipeline") as mock_pipeline,
        ):
            mock_vision_cls.return_value.analyze.return_value = _DEFAULT_VISION_STRUCTURE
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


def test_cli_llm_provider_flag_passed_to_generator(tmp_path: Path):
    """--llm-provider must reach LLMContentGenerator, and this must only happen
    once --pages truly exceeds the built-in content list."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with (
        patch("pageloom.cli.Pipeline") as mock_pipeline,
        patch("pageloom.cli.LLMContentGenerator") as mock_llm_gen_cls,
    ):
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "output.pdf"
        mock_llm_gen_cls.return_value.generate.return_value = ["Dragon", "Unicorn"]

        result = main(
            [
                "generate",
                "--pages",
                str(len(DEFAULT_CONTENTS) + 2),
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--use-llm",
                "--llm-provider",
                "openrouter",
            ]
        )
        assert result == 0
        mock_llm_gen_cls.assert_called_once_with(provider="openrouter")


def test_cli_use_llm_failure_returns_error():
    """When --pages exceeds the built-in list and no provider key is configured,
    the LLM call fails and the CLI reports a clear error (no Pipeline is run)."""
    result = main(["generate", "--pages", str(len(DEFAULT_CONTENTS) + 1), "--use-llm"])
    assert result == 1


def test_cli_no_subcommand():
    result = main([])
    assert result == 1


def test_cli_non_alphabet_structure_overflow_triggers_llm(tmp_path: Path):
    """Non-alphabet page structures (e.g. object_name_pair) have 0 built-in content,
    so any positive --pages exceeds available content and requires --use-llm."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="green")
    img.save(img_path)

    # Force structure detection to return a non-alphabet structure.
    with (
        patch(
            "pageloom.cli.detect_structure",
            return_value={
                "content_pattern": "object_name_pair",
                "element_count": 1,
                "arrangement": "centered",
                "text_image_relationship": "label_per_image",
            },
        ),
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
    ):
        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Cat, Dog"))]
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

                # Without --use-llm, any positive pages should fail (0 available content)
                result = main(
                    [
                        "generate",
                        "--pages",
                        "2",
                        "--example",
                        str(img_path),
                        "--output",
                        str(tmp_path / "out.pdf"),
                    ]
                )
                assert result == 1

                # With --use-llm, it should succeed by generating all content via LLM
                result_llm = main(
                    [
                        "generate",
                        "--pages",
                        "2",
                        "--example",
                        str(img_path),
                        "--output",
                        str(tmp_path / "out_llm.pdf"),
                        "--use-llm",
                    ]
                )
                assert result_llm == 0
                contents = mock_instance.run.call_args.kwargs.get("contents")
                assert contents == ["Cat", "Dog"]
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


def test_cli_non_alphabet_overflow_error_message(tmp_path: Path):
    """The overflow error for non-alphabet pages should report 0 available items."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="blue")
    img.save(img_path)

    with patch(
        "pageloom.cli.detect_structure",
        return_value={
            "content_pattern": "multi_item_scene",
            "element_count": 3,
            "arrangement": "grid",
            "text_image_relationship": "label_per_image",
        },
    ):
        result = main(
            [
                "generate",
                "--pages",
                "5",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
        assert result == 1


def test_cli_description_flag_reaches_classifier_and_pipeline(tmp_path: Path):
    """--description must be passed through to structure detection (so structural
    routing works from the CLI, not just the library API) and to Pipeline. Providing
    --description should also skip automatic vision analysis entirely (no network call)."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (800, 400), color="green")
    img.save(img_path)

    detected_structure = {
        "content_pattern": "object_name_pair",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
    }

    with (
        patch("pageloom.cli.detect_structure", return_value=detected_structure) as mock_detect,
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
        patch("pageloom.cli.LLMContentGenerator") as mock_llm_gen_cls,
    ):
        mock_llm_gen_cls.return_value.generate.return_value = ["Cat", "Dog"]
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
                "--description",
                "object name pair",
                "--output",
                str(tmp_path / "out.pdf"),
                "--use-llm",
            ]
        )
        assert result == 0
        mock_vision_cls.assert_not_called()
        mock_detect.assert_called_once_with(
            source=str(img_path), description="object name pair", vision_analyzer=None
        )
        mock_pipeline.assert_called_once_with(
            example_image=str(img_path),
            description="object name pair",
            page_structure=detected_structure,
            provider=None,
            model=None,
            vision_analyzer=None,
            verify_consistency=True,
            max_cost_usd=None,
            page_size=None,
        )


def test_cli_default_auto_vision_used_when_no_description(tmp_path: Path):
    """With no --description and no --no-vision, the CLI should construct a
    PageVisionAnalyzer and route its result (an open-ended structure, not one of
    the 3 legacy categories) all the way through to content generation and Pipeline."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (612, 612), color="orange")
    img.save(img_path)

    vision_structure = {
        "content_pattern": "number_count_matching",
        "content_type_description": (
            "a page pairing a printed number with a matching count of illustrated objects"
        ),
        "element_count": 3,
        "arrangement": "row",
        "text_image_relationship": "label_per_image",
        "example_content_items": ["3 ducks"],
    }

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
        patch("pageloom.cli.LLMContentGenerator") as mock_llm_gen_cls,
    ):
        mock_vision_cls.return_value.analyze.return_value = vision_structure
        mock_llm_gen_cls.return_value.generate.return_value = ["4 balloons", "5 stars"]
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
                "--use-llm",
            ]
        )
        assert result == 0
        mock_vision_cls.assert_called_once_with(provider=None)
        mock_vision_cls.return_value.analyze.assert_called_once_with(str(img_path), description="")
        # Novel, non-legacy content_pattern -> 0 built-in content -> LLM generates all of it.
        mock_llm_gen_cls.return_value.generate.assert_called_once_with(
            2, page_structure=vision_structure
        )
        contents = mock_instance.run.call_args.kwargs.get("contents")
        assert contents == ["4 balloons", "5 stars"]
        mock_pipeline.assert_called_once_with(
            example_image=str(img_path),
            description="",
            page_structure=vision_structure,
            provider=None,
            model=None,
            vision_analyzer=mock_vision_cls.return_value,
            verify_consistency=True,
            max_cost_usd=None,
            page_size=None,
        )


def test_cli_no_vision_flag_skips_vision_analysis(tmp_path: Path):
    """--no-vision must skip PageVisionAnalyzer construction entirely and fall
    back straight to heuristic/description-based structure detection."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
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
                "--no-vision",
            ]
        )
        assert result == 0
        mock_vision_cls.assert_not_called()


def test_cli_vision_unavailable_falls_back_gracefully(tmp_path: Path, capsys):
    """If PageVisionAnalyzer construction fails (e.g. no API key configured), the
    CLI should print a warning and continue via the heuristic fallback rather
    than crashing."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with (
        patch(
            "pageloom.cli.PageVisionAnalyzer",
            side_effect=VisionAnalysisError("no API key configured"),
        ),
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
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
        captured = capsys.readouterr()
        assert "vision analysis unavailable" in captured.err.lower()


def test_cli_vision_model_rejects_image_falls_back_gracefully(tmp_path: Path, capsys):
    """If the vision model rejects/ignores the image (e.g. provider exception),
    the CLI should exit 0, print the fallback warning, and complete generation
    via the heuristic path."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_instance = MagicMock()
        mock_vision_cls.return_value = mock_instance
        mock_instance.analyze.side_effect = VisionAnalysisError(
            "This model does not support image inputs"
        )
        mock_pipeline.return_value = mock_instance
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.run.return_value = tmp_path / "output.pdf"

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
        captured = capsys.readouterr()
        assert "vision-based page analysis failed" in captured.err
        assert "This model does not support image inputs" in captured.err
        mock_pipeline.assert_called_once()
        mock_pipeline_instance.run.assert_called_once()


def test_cli_check_vision_passes(tmp_path: Path):
    """check-vision should report PASS when vision provider and analysis work."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (256, 256), color="red").save(img_path)

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False),
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
    ):
        mock_instance = MagicMock()
        mock_vision_cls.return_value = mock_instance
        mock_instance.provider_name = "openrouter"
        mock_instance.provider = MagicMock()
        mock_instance.provider.model = "openrouter/free"
        mock_instance.provider.endpoint = None
        mock_instance.analyze.return_value = {
            "content_pattern": "digit_illustration",
            "element_count": 1,
            "arrangement": "centered",
            "text_image_relationship": "label_per_image",
        }
        result = main(["check-vision", "--example", str(img_path)])
        assert result == 0
        mock_vision_cls.assert_called_once_with(provider=None)
        mock_instance.analyze.assert_called_once()


def test_cli_check_vision_fails_on_missing_key():
    """check-vision should exit 1 and print FAIL when the provider has no API key."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "pageloom.cli.PageVisionAnalyzer",
            side_effect=VisionAnalysisError("no API key"),
        ),
    ):
        result = main(["check-vision"])
        assert result == 1


def test_cli_check_vision_reports_provider_and_model(tmp_path: Path, capsys):
    """check-vision should clearly report provider/model and smoke-test result."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (256, 256), color="blue").save(img_path)

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False),
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
    ):
        mock_instance = MagicMock()
        mock_vision_cls.return_value = mock_instance
        mock_instance.provider_name = "openrouter"
        mock_instance.provider = MagicMock()
        mock_instance.provider.model = "openrouter/free"
        mock_instance.provider.endpoint = "https://openrouter.ai/api/v1"
        mock_instance.analyze.return_value = {
            "content_pattern": "test",
            "element_count": 2,
            "arrangement": "row",
            "text_image_relationship": "label_per_image",
        }
        result = main(
            ["check-vision", "--vision-provider", "openrouter", "--example", str(img_path)]
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "openrouter" in captured.out
        assert "openrouter/free" in captured.out
        assert "PASS" in captured.out


def test_cli_check_vision_analysis_failure(tmp_path: Path):
    """If analyze() raises VisionAnalysisError, check-vision exits 1 with FAIL."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (256, 256), color="green").save(img_path)

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False),
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
    ):
        mock_instance = MagicMock()
        mock_vision_cls.return_value = mock_instance
        mock_instance.provider_name = "openrouter"
        mock_instance.provider = MagicMock()
        mock_instance.provider.model = "openrouter/free"
        mock_instance.provider.endpoint = None
        mock_instance.analyze.side_effect = VisionAnalysisError("bad response")
        result = main(["check-vision", "--example", str(img_path)])
        assert result == 1


def test_cli_check_image_passes(tmp_path: Path, capsys):
    """check-image should report PASS when the provider generates an image."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pageloom.cli.build_image_provider") as mock_build,
    ):
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"
        mock_provider.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        mock_provider.generate.return_value = Image.new("RGB", (256, 256), color="red")
        mock_build.return_value = mock_provider

        result = main(["check-image"])
        assert result == 0
        captured = capsys.readouterr()
        assert "local" in captured.out
        assert "stabilityai/stable-diffusion-xl-base-1.0" in captured.out
        assert "PASS" in captured.out
        mock_build.assert_called_once_with(provider=None, model=None)
        mock_provider.generate.assert_called_once()


def test_cli_check_image_fails_on_setup():
    """check-image should exit 1 and print FAIL when provider setup fails."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pageloom.cli.build_image_provider", side_effect=RuntimeError("no API key")),
    ):
        result = main(["check-image"])
        assert result == 1


def test_cli_check_image_reports_provider_and_model(tmp_path: Path, capsys):
    """check-image should clearly report provider/model and smoke-test result."""
    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False),
        patch("pageloom.cli.build_image_provider") as mock_build,
    ):
        mock_provider = MagicMock()
        mock_provider.provider_name = "gemini"
        mock_provider.model_id = "gemini-nano-banana-2.1"
        mock_provider.generate.return_value = Image.new("RGB", (256, 256), color="blue")
        mock_build.return_value = mock_provider
        result = main(
            ["check-image", "--image-provider", "gemini", "--image-model", "gemini-nano-banana-2.1"]
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "gemini" in captured.out
        assert "gemini-nano-banana-2.1" in captured.out
        assert "PASS" in captured.out


def test_cli_check_image_generation_failure(tmp_path: Path, capsys):
    """If generate() raises, check-image exits 1 with FAIL naming the error."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pageloom.cli.build_image_provider") as mock_build,
    ):
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"
        mock_provider.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        mock_provider.generate.side_effect = RuntimeError("model download failed")
        mock_build.return_value = mock_provider
        result = main(["check-image"])
        assert result == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert "model download failed" in captured.err


def test_cli_vision_provider_flag_passed_through(tmp_path: Path):
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = {
            "content_pattern": "single_letter_word_illustration",
            "content_type_description": "a letter and word page",
            "element_count": 1,
            "arrangement": "centered",
            "text_image_relationship": "label_per_image",
            "example_content_items": ["Apple"],
        }
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
                "--vision-provider",
                "anthropic",
            ]
        )
        assert result == 0
        mock_vision_cls.assert_called_once_with(provider="anthropic")


def test_cli_image_provider_and_model_flags_passed_through(tmp_path: Path):
    """`--image-provider` and `--image-model` are passed through to Pipeline
    so provider switching works through the CLI."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = dict(_DEFAULT_VISION_STRUCTURE)
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "3",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--image-provider",
                "local",
                "--image-model",
                "custom/image-model",
            ]
        )
        assert result == 0
        mock_pipeline.assert_called_once()
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["provider"] == "local"
        assert kwargs["model"] == "custom/image-model"


def _cli_generate_with_output_format(output_format: str, tmp_path: Path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = dict(_DEFAULT_VISION_STRUCTURE)
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"
        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--output-format",
                output_format,
            ]
        )
        assert result == 0
        return mock_instance.run.call_args.kwargs.get("output_format")


def test_cli_output_format_pdf_passes_to_pipeline(tmp_path: Path):
    assert _cli_generate_with_output_format("pdf", tmp_path) == "pdf"


def test_cli_output_format_png_passes_to_pipeline(tmp_path: Path):
    assert _cli_generate_with_output_format("png", tmp_path) == "png"


def test_cli_output_format_jpeg_passes_to_pipeline(tmp_path: Path):
    assert _cli_generate_with_output_format("jpeg", tmp_path) == "jpeg"


def test_cli_output_format_zip_passes_to_pipeline(tmp_path: Path):
    assert _cli_generate_with_output_format("zip", tmp_path) == "zip"


def test_cli_output_format_default_is_pdf(tmp_path: Path):
    kwargs = _capture_run_kwargs(tmp_path)
    assert kwargs.get("output_format") == "pdf"


def _capture_run_kwargs(tmp_path: Path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = dict(_DEFAULT_VISION_STRUCTURE)
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"
        main(
            [
                "generate",
                "--pages",
                "1",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
        return mock_instance.run.call_args.kwargs


def test_cli_invalid_output_format_is_rejected(tmp_path: Path):
    img_path = tmp_path / "example.png"
    Image.new("RGB", (512, 512), color="purple").save(img_path)
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "generate",
                "--pages",
                "1",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--output-format",
                "bmp",
            ]
        )
    assert exc_info.value.code == 2


def test_cli_generate_runs_consistency_guard_with_vision_analyzer(tmp_path: Path, capsys):
    """`pageloom generate` must actually run ConsistencyGuard on every generated
    page (previously the class was only ever constructed inside tests), and the
    guard must receive the same vision_analyzer the CLI used to classify the
    example so generated pages are re-classified with the same method."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="purple").save(img_path)

    vision_structure = {
        "content_pattern": "single_letter_word_illustration",
        "content_type_description": "a letter and word page",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
        "example_content_items": ["Apple"],
    }

    captured: dict = {}
    verify_calls: list = []

    def make_guard(*args, **kwargs):
        guard = ConsistencyGuard(*args, **kwargs)
        captured["guard"] = guard
        captured["kwargs"] = kwargs
        original_verify = guard.verify

        def spy_verify(generated_image):
            verify_calls.append(generated_image)
            return original_verify(generated_image)

        guard.verify = spy_verify  # type: ignore[method-assign]
        return guard

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.pipeline.build_image_provider") as mock_gen_cls,
        patch("pageloom.pipeline.ConsistencyGuard", side_effect=make_guard),
    ):
        mock_vision_cls.return_value.analyze.return_value = vision_structure
        mock_gen_cls.return_value.generate.side_effect = lambda prompt, width, height: (
            _passing_image(width, height)
        )

        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
            ]
        )
        assert result == 0

    # The guard was constructed with the CLI's vision analyzer: generated pages
    # are re-classified with the same method used for the example.
    assert captured["kwargs"]["vision_analyzer"] is mock_vision_cls.return_value
    assert captured["guard"].vision_analyzer is mock_vision_cls.return_value
    # verify() ran once per generated page during `pageloom generate`.
    assert len(verify_calls) == 2
    # And the run reported the check to the user.
    out = capsys.readouterr()
    assert "Consistency check: " in out.out
    assert " page(s) match the example." in out.out
    assert (tmp_path / "out.pdf").exists()


def test_cli_no_consistency_flag_skips_check(tmp_path: Path, capsys):
    """--no-consistency must pass verify_consistency=False to Pipeline, so no
    guard runs (no extra vision API call per page) and no `Consistency check:`
    summary is printed -- mirroring what --no-vision does for example analysis."""
    img_path = tmp_path / "example.png"
    Image.new("RGB", (612, 612), color="purple").save(img_path)

    vision_structure = {
        "content_pattern": "single_letter_word_illustration",
        "content_type_description": "a letter and word page",
        "element_count": 1,
        "arrangement": "centered",
        "text_image_relationship": "label_per_image",
        "example_content_items": ["Apple"],
    }

    with (
        patch("pageloom.cli.PageVisionAnalyzer") as mock_vision_cls,
        patch("pageloom.cli.Pipeline") as mock_pipeline,
    ):
        mock_vision_cls.return_value.analyze.return_value = vision_structure
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance
        mock_instance.run.return_value = tmp_path / "out.pdf"

        result = main(
            [
                "generate",
                "--pages",
                "2",
                "--example",
                str(img_path),
                "--output",
                str(tmp_path / "out.pdf"),
                "--no-consistency",
            ]
        )
        assert result == 0

    # The CLI explicitly opts the pipeline out of verification.
    assert mock_pipeline.call_args.kwargs["verify_consistency"] is False
    assert mock_pipeline.call_args.kwargs["vision_analyzer"] is mock_vision_cls.return_value
    # And nothing produced a `Consistency check:` line for the user.
    out = capsys.readouterr()
    assert "Consistency check:" not in out.out


def test_cli_images_with_prompts(tmp_path: Path, capsys):
    """`pageloom images` renders one image per positional prompt."""
    with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
        mock_renderer = MagicMock()
        mock_renderer.estimated_total_cost_usd.return_value = 0.0
        mock_renderer.render.return_value = [
            Image.new("RGB", (64, 64), color="red"),
            Image.new("RGB", (64, 64), color="blue"),
        ]
        mock_renderer_cls.return_value = mock_renderer

        result = main(
            [
                "images",
                "a red ball",
                "a blue cube",
                "--output",
                str(tmp_path / "out.png"),
            ]
        )
        assert result == 0
        kwargs = mock_renderer_cls.call_args.kwargs
        assert kwargs["contents"] == ["a red ball", "a blue cube"]
        mock_renderer.render.assert_called_once()

    out = capsys.readouterr()
    assert "Generated 2 image(s)" in out.out
    assert (tmp_path / "out_001.png").exists()
    assert (tmp_path / "out_002.png").exists()


def test_cli_images_content_file_txt(tmp_path: Path, capsys):
    """`--content-file` supplies prompts from a plain-text file."""
    content_file = tmp_path / "prompts.txt"
    content_file.write_text("one\ntwo\n", encoding="utf-8")

    with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
        mock_renderer = MagicMock()
        mock_renderer.estimated_total_cost_usd.return_value = 0.0
        mock_renderer.render.return_value = [Image.new("RGB", (64, 64), color="red")]
        mock_renderer_cls.return_value = mock_renderer

        result = main(
            ["images", "--content-file", str(content_file), "--output", str(tmp_path / "out.png")]
        )
        assert result == 0
        assert mock_renderer_cls.call_args.kwargs["contents"] == ["one", "two"]


def test_cli_images_content_file_json(tmp_path: Path, capsys):
    """`--content-file` supports JSON lists of strings and prompt objects."""
    content_file = tmp_path / "prompts.json"
    content_file.write_text('[{"prompt": "cat"}, "dog"]', encoding="utf-8")

    with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
        mock_renderer = MagicMock()
        mock_renderer.estimated_total_cost_usd.return_value = 0.0
        mock_renderer.render.return_value = [Image.new("RGB", (64, 64), color="red")]
        mock_renderer_cls.return_value = mock_renderer

        result = main(
            ["images", "--content-file", str(content_file), "--output", str(tmp_path / "out.png")]
        )
        assert result == 0
        assert mock_renderer_cls.call_args.kwargs["contents"] == ["cat", "dog"]


def test_cli_images_no_prompts_errors(capsys):
    """No prompts and no content file must exit 1 with a clear error."""
    result = main(["images"])
    assert result == 1
    assert "no prompts given" in capsys.readouterr().err


def test_cli_images_output_formats(tmp_path: Path):
    """png, jpeg and zip output formats all write the generated images."""
    for fmt, expected_ext in [("png", "png"), ("jpeg", "jpg")]:
        with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
            mock_renderer = MagicMock()
            mock_renderer.estimated_total_cost_usd.return_value = 0.0
            mock_renderer.render.return_value = [
                Image.new("RGB", (64, 64), color="red"),
                Image.new("RGB", (64, 64), color="blue"),
            ]
            mock_renderer_cls.return_value = mock_renderer

            result = main(
                [
                    "images",
                    "a",
                    "b",
                    "--output-format",
                    fmt,
                    "--output",
                    str(tmp_path / f"out_{fmt}"),
                ]
            )
            assert result == 0
        assert (tmp_path / f"out_{fmt}_001.{expected_ext}").exists()
        assert (tmp_path / f"out_{fmt}_002.{expected_ext}").exists()


def test_cli_images_zip_output(tmp_path: Path):
    """The zip output format bundles numbered PNGs into one archive."""
    zip_path = tmp_path / "out.zip"
    with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
        mock_renderer = MagicMock()
        mock_renderer.estimated_total_cost_usd.return_value = 0.0
        mock_renderer.render.return_value = [
            Image.new("RGB", (64, 64), color="red"),
        ]
        mock_renderer_cls.return_value = mock_renderer

        result = main(["images", "a", "--output-format", "zip", "--output", str(zip_path)])
        assert result == 0

    import zipfile

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert "001.png" in archive.namelist()


def test_cli_images_style_reference_forwarded(tmp_path: Path, capsys):
    """--style-reference is handed to the renderer unchanged."""
    reference = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), color="green").save(reference)

    with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
        mock_renderer = MagicMock()
        mock_renderer.estimated_total_cost_usd.return_value = 0.0
        mock_renderer.render.return_value = [Image.new("RGB", (64, 64), color="red")]
        mock_renderer_cls.return_value = mock_renderer

        result = main(["images", "a prompt", "--style-reference", str(reference)])
        assert result == 0
        assert mock_renderer_cls.call_args.kwargs["style_reference"] == str(reference)


def test_cli_images_provider_and_model_flags_passed_through(tmp_path: Path, capsys):
    """--image-provider and --image-model reach the renderer."""
    with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
        mock_renderer = MagicMock()
        mock_renderer.estimated_total_cost_usd.return_value = 0.0
        mock_renderer.render.return_value = [Image.new("RGB", (64, 64), color="red")]
        mock_renderer_cls.return_value = mock_renderer

        result = main(
            [
                "images",
                "a prompt",
                "--image-provider",
                "gemini",
                "--image-model",
                "gemini-nano-banana-2.1",
            ]
        )
        assert result == 0
        kwargs = mock_renderer_cls.call_args.kwargs
        assert kwargs["provider_name"] == "gemini"
        assert kwargs["model"] == "gemini-nano-banana-2.1"


def test_cli_images_estimate_prints_cost_without_rendering(tmp_path: Path, capsys):
    """--estimate prints the expected cost and never calls render()."""
    with patch("pageloom.cli.ImageSetRenderer") as mock_renderer_cls:
        mock_renderer = MagicMock()
        mock_renderer.estimated_total_cost_usd.return_value = 5.25
        mock_renderer_cls.return_value = mock_renderer

        result = main(["images", "a prompt", "--estimate"])
        assert result == 0
        mock_renderer.render.assert_not_called()

    assert "$5.25" in capsys.readouterr().out
