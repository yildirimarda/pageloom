import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from pageloom.pdf_compiler import (
    DEFAULT_PAGE_SIZE,
    PDFCompiler,
    parse_page_size,
    write_individual_images,
    write_zip,
)


def test_default_page_size_is_square_612():
    assert DEFAULT_PAGE_SIZE == (612.0, 612.0)


def test_kdp_page_size_deprecated_alias_emits_warning():
    """KDP_PAGE_SIZE is kept as a deprecated alias for one release."""
    import warnings

    from pageloom import pdf_compiler

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = pdf_compiler.KDP_PAGE_SIZE
    assert alias == DEFAULT_PAGE_SIZE
    assert any(
        issubclass(w.category, DeprecationWarning) and "KDP_PAGE_SIZE" in str(w.message)
        for w in caught
    )


def test_pdf_compiler_init_default_size():
    compiler = PDFCompiler()
    assert compiler.page_size == (612.0, 612.0)


def test_pdf_compiler_init_custom_size():
    compiler = PDFCompiler(page_size=(800.0, 600.0))
    assert compiler.page_size == (800.0, 600.0)


def test_compile_single_image(tmp_path):
    output = tmp_path / "output.pdf"
    img = Image.new("RGB", (400, 400), color="white")
    compiler = PDFCompiler()
    result = compiler.compile([img], output)
    assert result.exists()
    assert result == output


def test_compile_multiple_images(tmp_path):
    output = tmp_path / "multi.pdf"
    img1 = Image.new("RGB", (300, 300), color="red")
    img2 = Image.new("RGB", (300, 300), color="blue")
    compiler = PDFCompiler()
    result = compiler.compile([img1, img2], output)
    assert result.exists()


def test_compile_from_file_path(tmp_path):
    output = tmp_path / "from_path.pdf"
    img_path = tmp_path / "page.png"
    img = Image.new("RGB", (200, 200), color="green")
    img.save(img_path)
    compiler = PDFCompiler()
    result = compiler.compile([img_path], output)
    assert result.exists()


def test_compile_mixed_image_and_path(tmp_path):
    output = tmp_path / "mixed.pdf"
    img_path = tmp_path / "page.png"
    img = Image.new("RGB", (200, 200), color="green")
    img.save(img_path)
    compiler = PDFCompiler()
    result = compiler.compile([img, img_path], output)
    assert result.exists()


def test_compile_creates_output_directory(tmp_path):
    output = tmp_path / "nested" / "subdir" / "output.pdf"
    img = Image.new("RGB", (100, 100), color="black")
    compiler = PDFCompiler()
    result = compiler.compile([img], output)
    assert result.exists()
    assert output.parent.exists()


def test_compile_output_has_correct_page_count(tmp_path):
    from pypdf import PdfReader

    output = tmp_path / "pages.pdf"
    img1 = Image.new("RGB", (300, 300), color="red")
    img2 = Image.new("RGB", (300, 300), color="blue")
    compiler = PDFCompiler()
    compiler.compile([img1, img2], output)
    reader = PdfReader(str(output))
    assert len(reader.pages) == 2


def test_compile_unsupported_type_raises(tmp_path):
    output = tmp_path / "bad.pdf"
    compiler = PDFCompiler()
    try:
        compiler.compile([42], output)
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_parse_page_size_none_returns_none():
    assert parse_page_size(None) is None


def test_parse_page_size_presets():
    assert parse_page_size("square") == DEFAULT_PAGE_SIZE
    assert parse_page_size("a4") == (595.28, 841.89)
    assert parse_page_size("letter") == (612.0, 792.0)


def test_parse_page_size_presets_case_insensitive():
    assert parse_page_size("SQUARE") == DEFAULT_PAGE_SIZE
    assert parse_page_size(" A4 ") == (595.28, 841.89)


def test_parse_page_size_custom_widthxheight():
    assert parse_page_size("612x792") == (612.0, 792.0)
    assert parse_page_size("800 x 600") == (800.0, 600.0)


def test_parse_page_size_invalid_raises():
    with pytest.raises(ValueError):
        parse_page_size("not-a-size")


def _make_pages(tmp_path: Path, count: int = 3) -> list[Image.Image]:
    return [Image.new("RGB", (100, 100), color=c) for c in ("red", "green", "blue")][:count]


def test_write_individual_images_png(tmp_path: Path):
    output = tmp_path / "out.png"
    pages = _make_pages(tmp_path)
    result = write_individual_images(pages, output, fmt="png")
    assert result == tmp_path
    for index in range(1, len(pages) + 1):
        file_path = tmp_path / f"out_{index:03d}.png"
        assert file_path.exists()
        with Image.open(file_path) as img:
            assert img.format == "PNG"
            assert img.size == (100, 100)


def test_write_individual_images_jpeg(tmp_path: Path):
    output = tmp_path / "out.jpeg"
    pages = _make_pages(tmp_path)
    result = write_individual_images(pages, output, fmt="jpeg")
    assert result == tmp_path
    for index in range(1, len(pages) + 1):
        file_path = tmp_path / f"out_{index:03d}.jpg"
        assert file_path.exists()
        with Image.open(file_path) as img:
            assert img.format == "JPEG"
            assert img.mode == "RGB"


def test_write_individual_images_creates_directory(tmp_path: Path):
    output = tmp_path / "nested" / "out.png"
    pages = _make_pages(tmp_path, 2)
    write_individual_images(pages, output, fmt="png")
    assert (tmp_path / "nested" / "out_001.png").exists()
    assert (tmp_path / "nested" / "out_002.png").exists()


def test_write_individual_images_invalid_format_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        write_individual_images(_make_pages(tmp_path, 1), tmp_path / "out.png", fmt="bmp")


def test_write_zip(tmp_path: Path):
    output = tmp_path / "out.zip"
    pages = _make_pages(tmp_path)
    result = write_zip(pages, output)
    assert result == output
    assert output.exists()
    with zipfile.ZipFile(str(output)) as archive:
        names = archive.namelist()
        assert names == ["001.png", "002.png", "003.png"]
        for index in range(1, len(pages) + 1):
            data = archive.read(f"{index:03d}.png")
            with Image.open(io.BytesIO(data)) as img:
                assert img.format == "PNG"
                assert img.mode == "RGB"
