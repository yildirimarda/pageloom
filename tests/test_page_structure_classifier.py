from pathlib import Path

from PIL import Image

from pageloom.page_structure_classifier import PageStructureClassifier


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
    classifier = PageStructureClassifier(
        source=img, description="multi-item labeled scene"
    )
    result = classifier.classify()
    assert result["content_pattern"] == "multi_item_scene"
    assert result["arrangement"] == "grid"
    assert result["element_count"] >= 2


def test_classifier_classify_matching_puzzle_grid():
    img = Image.new("RGB", (800, 400), color="yellow")
    classifier = PageStructureClassifier(
        source=img, description="matching puzzle grid page"
    )
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
