from pathlib import Path

from PIL import Image

from pageloom.style_extractor import VisualStyleExtractor


def test_extractor_init_from_image():
    img = Image.new("RGB", (512, 512), color="blue")
    extractor = VisualStyleExtractor(img)
    assert extractor.image is not None


def test_extractor_init_from_path(tmp_path: Path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (300, 300), color="green")
    img.save(img_path)
    extractor = VisualStyleExtractor(str(img_path))
    assert extractor.image.size == (300, 300)


def test_extractor_init_bad_type():
    try:
        VisualStyleExtractor(123)  # type: ignore[arg-type]
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_extract_palette():
    img = Image.new("RGB", (200, 200), color="red")
    extractor = VisualStyleExtractor(img)
    palette = extractor.extract_palette()
    assert isinstance(palette, list)
    assert len(palette) > 0
    assert all(isinstance(c, str) for c in palette)
    assert palette[0].startswith("rgb(")


def test_extract_typography():
    img = Image.new("RGB", (400, 400), color="yellow")
    extractor = VisualStyleExtractor(img)
    result = extractor.extract_typography()
    assert isinstance(result, dict)
    assert "letter_color" in result
    assert "word_color" in result
    assert "font_family_hint" in result
    assert "letter_size_estimate" in result


def test_extract_composition():
    img = Image.new("RGB", (600, 600), color="white")
    extractor = VisualStyleExtractor(img)
    result = extractor.extract_composition()
    assert isinstance(result, dict)
    assert "margins" in result
    assert "layout_type" in result
    assert "page_dimensions" in result
    assert result["page_dimensions"]["width"] == 600


def test_extract_full():
    img = Image.new("RGB", (512, 512), color="purple")
    extractor = VisualStyleExtractor(img)
    result = extractor.extract()
    assert "palette" in result
    assert "typography" in result
    assert "composition" in result
    assert isinstance(result["palette"], list)


def test_extract_structure_uses_description():
    img = Image.new("RGB", (800, 400), color="green")
    extractor = VisualStyleExtractor(img, description="object name pair")
    structure = extractor.extract_structure()
    assert structure.get("content_pattern") == "object_name_pair"


def test_extract_structural_attributes():
    img = Image.new("RGB", (512, 512), color="blue")
    extractor = VisualStyleExtractor(img)
    result = extractor.extract()
    assert "structural_attributes" in result
    attrs = result["structural_attributes"]
    assert "element_count" in attrs
    assert isinstance(attrs["element_count"], int)
    assert "arrangement" in attrs
    assert "text_image_relationship" in attrs
