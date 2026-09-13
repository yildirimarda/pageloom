import importlib.util
import pathlib
from unittest.mock import MagicMock, patch

PACKAGE_DIR = pathlib.Path("src/pageloom")


def test_generator_module_importable():
    spec = importlib.util.spec_from_file_location(
        "pageloom.generator", PACKAGE_DIR / "generator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "IllustrationGenerator")
    assert hasattr(module, "StableDiffusionPipeline")


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=False)
def test_illustration_generator_init(mock_mps, mock_cuda, mock_from_pretrained):
    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator()
    assert gen.device == "cpu"
    mock_from_pretrained.assert_called_once()
    mock_pipeline.to.assert_called_once_with("cpu")


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=True)
def test_illustration_generator_init_cuda(mock_cuda, mock_from_pretrained):
    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator()
    assert gen.device == "cuda"


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=True)
def test_illustration_generator_init_mps(mock_cuda, mock_mps, mock_from_pretrained):
    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator()
    assert gen.device == "mps"


def test_generator_uses_identical_settings_on_cpu_cuda_and_mps():
    from unittest.mock import MagicMock, patch

    from pageloom.generator import IllustrationGenerator

    # Verify the same torch_dtype and attention slicing regardless of device
    with patch(
        "pageloom.generator.StableDiffusionPipeline.from_pretrained"
    ) as mock_from_pretrained:
        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = lambda device: mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        # CPU path
        with (
            patch("pageloom.generator.torch.cuda.is_available", return_value=False),
            patch("pageloom.generator.torch.backends.mps.is_available", return_value=False),
        ):
            gen_cpu = IllustrationGenerator()
        # CUDA path
        with patch("pageloom.generator.torch.cuda.is_available", return_value=True):
            gen_cuda = IllustrationGenerator()
        # MPS path
        with (
            patch("pageloom.generator.torch.cuda.is_available", return_value=False),
            patch("pageloom.generator.torch.backends.mps.is_available", return_value=True),
        ):
            gen_mps = IllustrationGenerator()

    # All three should use float32 (no behavioral difference)
    assert gen_cpu.torch_dtype == gen_cuda.torch_dtype == gen_mps.torch_dtype
    assert str(gen_cpu.torch_dtype) == str(gen_cuda.torch_dtype) == str(gen_mps.torch_dtype)
    # Attention slicing should be enabled consistently
    assert mock_pipeline.enable_attention_slicing.call_count == 3


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=True)
def test_illustration_generator_generate_mps(mock_mps, mock_cuda, mock_from_pretrained):
    from PIL import Image

    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_image = Image.new("RGB", (512, 512), color="red")
    mock_pipeline.return_value = MagicMock(images=[mock_image])
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator()
    assert gen.device == "mps"
    result = gen.generate("a happy cat")
    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=False)
def test_illustration_generator_generate(mock_cuda, mock_mps, mock_from_pretrained):
    from PIL import Image

    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_image = Image.new("RGB", (512, 512), color="red")
    mock_pipeline.return_value = MagicMock(images=[mock_image])
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator()
    result = gen.generate("a happy cat")
    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=False)
def test_illustration_generator_generate_empty_images(mock_cuda, mock_mps, mock_from_pretrained):
    from PIL import Image

    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_pipeline.return_value = MagicMock(images=[])
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator()
    result = gen.generate("a happy cat", width=256, height=128)
    assert isinstance(result, Image.Image)
    assert result.size == (256, 128)
    assert result.getpixel((0, 0)) == (128, 128, 128)


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=False)
def test_illustration_generator_generate_no_images_attr(mock_cuda, mock_mps, mock_from_pretrained):
    from PIL import Image

    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_result = MagicMock()
    del mock_result.images
    mock_pipeline.return_value = mock_result
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator()
    result = gen.generate("a happy cat", width=128, height=64)
    assert isinstance(result, Image.Image)
    assert result.size == (128, 64)


def test_illustration_generator_init_license_check_raises_for_unknown_model():
    from pageloom.generator import IllustrationGenerator
    from pageloom.license_check import LicenseVerificationError

    try:
        IllustrationGenerator(
            model_id="unknown/model-id-123",
            verify_license=True,
        )
        assert False, "Expected LicenseVerificationError"
    except LicenseVerificationError as exc:
        assert "Unknown model" in str(exc)
        assert "book intended for sale" in str(exc)


def test_illustration_generator_init_skips_license_check():
    with (
        patch("pageloom.generator.torch.cuda.is_available", return_value=False),
        patch("pageloom.generator.torch.backends.mps.is_available", return_value=False),
        patch("pageloom.generator.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = lambda device: mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline
        from pageloom.generator import IllustrationGenerator

        # Unknown model but license verification disabled
        gen = IllustrationGenerator(model_id="unknown/model", verify_license=False)
        assert gen.verify_license is False


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=False)
def test_illustration_generator_init_model_load_failure(mock_mps, mock_cuda, mock_from_pretrained):
    mock_from_pretrained.side_effect = RuntimeError("model not found")
    from pageloom.generator import IllustrationGenerator

    try:
        IllustrationGenerator()
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Failed to load Stable Diffusion model" in str(exc)
        assert "model not found" in str(exc)
