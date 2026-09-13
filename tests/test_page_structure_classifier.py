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
