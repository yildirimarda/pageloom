"""Tests for the pageloom CLI entrypoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pageloom.cli import DEFAULT_CONTENTS, main
from pageloom.vision import VisionAnalysisError


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
            example_image=str(img_path), description="", page_structure=detected_structure
        )
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


def test_cli_pages_within_full_built_in_list_never_calls_llm(tmp_path: Path):
    """Requesting up to the full built-in content list (35 items) should use the
    built-in content directly and must NOT invoke the LLM generator, even when
    --use-llm is passed -- the overflow trigger is the true content-availability
    boundary (len(DEFAULT_CONTENTS)), not the legacy 26-letter alphabet count."""
    img_path = tmp_path / "example.png"
    img = Image.new("RGB", (512, 512), color="purple")
    img.save(img_path)

    with (
        patch("pageloom.cli.Pipeline") as mock_pipeline,
        patch("pageloom.cli.LLMContentGenerator") as mock_llm_gen,
    ):
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
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("pageloom.cli.Pipeline") as mock_pipeline,
        ):
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
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("pageloom.cli.Pipeline") as mock_pipeline,
        ):
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
            example_image=str(img_path), description="", page_structure=vision_structure
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
