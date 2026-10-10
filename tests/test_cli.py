"""Tests for the pageloom CLI entrypoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pageloom.cli import DEFAULT_CONTENTS, main
from pageloom.consistency import ConsistencyGuard
from pageloom.vision import VisionAnalysisError

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
            vision_analyzer=mock_vision_cls.return_value,
            verify_consistency=True,
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
            vision_analyzer=None,
            verify_consistency=True,
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
            vision_analyzer=mock_vision_cls.return_value,
            verify_consistency=True,
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
        patch("pageloom.pipeline.IllustrationGenerator") as mock_gen_cls,
        patch("pageloom.pipeline.ConsistencyGuard", side_effect=make_guard),
    ):
        mock_vision_cls.return_value.analyze.return_value = vision_structure
        mock_gen_cls.return_value.generate.side_effect = lambda prompt, width, height: Image.new(
            "RGB", (width, height), color="white"
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
