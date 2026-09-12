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
