from PIL import Image

from pageloom.pdf_compiler import KDP_PAGE_SIZE, PDFCompiler


def test_kdp_page_size_is_square_612():
    assert KDP_PAGE_SIZE == (612.0, 612.0)


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


def test_compile_unsupported_type_raises(tmp_path):
    output = tmp_path / "bad.pdf"
    compiler = PDFCompiler()
    try:
        compiler.compile([42], output)
        assert False, "Expected TypeError"
    except TypeError:
        pass
