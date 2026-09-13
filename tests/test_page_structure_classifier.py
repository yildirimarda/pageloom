from pathlib import Path

from PIL import Image

from pageloom.page_structure_classifier import PageStructureClassifier, detect_structure
from pageloom.vision import VisionAnalysisError


def test_classifier_init_from_path(tmp_path: Path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (300, 300), color="green")
    img.save(img_path)
    classifier = PageStructureClassifier(source=str(img_path), description="letter")
    assert classifier.image is not None


def test_classifier_classify_source():
    img = Image.new("RGB", (512, 512), color="blue")
    classifier = PageStructureClassifier(source=img, description="single letter word")
    result = classifier.classify()
    assert result["content_pattern"] == "single_letter_word_illustration"
    assert result["element_count"] == 1
    assert result["arrangement"] == "centered"


def test_classifier_classify_object_name_pair():
    img = Image.new("RGB", (512, 512), color="green")
    classifier = PageStructureClassifier(source=img, description="object name pair")
    result = classifier.classify()
    assert result["content_pattern"] == "object_name_pair"
    assert result["element_count"] == 1
    assert result["arrangement"] == "centered"
    assert result["text_image_relationship"] == "label_per_image"


def test_classifier_classify_multi_item_scene_from_description():
    img = Image.new("RGB", (612, 612), color="blue")
    classifier = PageStructureClassifier(source=img, description="multi-item labeled scene")
    result = classifier.classify()
    assert result["content_pattern"] == "multi_item_scene"
    assert result["arrangement"] == "grid"
    assert result["element_count"] >= 2


def test_classifier_classify_matching_puzzle_grid():
    img = Image.new("RGB", (800, 400), color="yellow")
    classifier = PageStructureClassifier(source=img, description="matching puzzle grid page")
    result = classifier.classify()
    assert result["content_pattern"] == "multi_item_scene"
    assert result["arrangement"] == "grid"


def test_classifier_classify_pure_image_wide():
    img = Image.new("RGB", (900, 400), color="red")
    classifier = PageStructureClassifier(source=img, description="")
    result = classifier.classify()
    assert result["content_pattern"] == "multi_item_scene"


def test_classifier_classify_different_image():
    img = Image.new("RGB", (512, 512), color="red")
    classifier = PageStructureClassifier(source=img, description="object name pair")
    other = Image.new("RGB", (400, 400), color="yellow")
    result = classifier.classify(other)
    # Pure image inference for non-source image
    assert "content_pattern" in result
    assert result["content_pattern"] in (
        "object_name_pair",
        "multi_item_scene",
        "single_letter_word_illustration",
    )


class _FakeVisionAnalyzer:
    def __init__(self, structure=None, error=None):
        self.structure = structure
        self.error = error
        self.calls = []

    def analyze(self, source, description=""):
        self.calls.append((source, description))
        if self.error is not None:
            raise self.error
        return self.structure


def test_detect_structure_prefers_vision_when_available():
    vision_structure = {
        "content_pattern": "number_count_matching",
        "content_type_description": "a page pairing a number with that many objects",
        "element_count": 3,
        "arrangement": "row",
        "text_image_relationship": "label_per_image",
        "example_content_items": ["3 ducks"],
    }
    analyzer = _FakeVisionAnalyzer(structure=vision_structure)
    img = Image.new("RGB", (612, 612), color="blue")
    result = detect_structure(source=img, description="", vision_analyzer=analyzer)
    assert result == vision_structure
    assert len(analyzer.calls) == 1


def test_detect_structure_falls_back_to_heuristic_on_vision_failure(capsys):
    analyzer = _FakeVisionAnalyzer(error=VisionAnalysisError("rate limited"))
    img = Image.new("RGB", (512, 512), color="blue")
    result = detect_structure(
        source=img, description="single letter word", vision_analyzer=analyzer
    )
    assert result["content_pattern"] == "single_letter_word_illustration"
    captured = capsys.readouterr()
    assert "falling back to" in captured.err.lower()


def test_detect_structure_vision_capability_hint_on_image_rejection(capsys):
    analyzer = _FakeVisionAnalyzer(
        error=VisionAnalysisError("This model does not support image inputs")
    )
    img = Image.new("RGB", (512, 512), color="blue")
    result = detect_structure(source=img, description="", vision_analyzer=analyzer)
    assert result["content_pattern"] == "single_letter_word_illustration"
    captured = capsys.readouterr()
    assert "vision-based page analysis failed" in captured.err
    assert "The selected model/provider does not appear to support image input" in captured.err


def test_detect_structure_no_extra_hint_on_rate_limit(capsys):
    analyzer = _FakeVisionAnalyzer(
        error=VisionAnalysisError("LLM provider 'openrouter' rate limit exceeded.")
    )
    img = Image.new("RGB", (512, 512), color="blue")
    result = detect_structure(source=img, description="", vision_analyzer=analyzer)
    assert result["content_pattern"] == "single_letter_word_illustration"
    captured = capsys.readouterr()
    assert "vision-based page analysis failed" in captured.err
    assert "does not appear to support image input" not in captured.err


def test_detect_structure_without_vision_analyzer_uses_heuristic():
    img = Image.new("RGB", (512, 512), color="blue")
    result = detect_structure(source=img, description="object name pair", vision_analyzer=None)
    assert result["content_pattern"] == "object_name_pair"
