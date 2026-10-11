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


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=False)
def test_production_generator_disables_safety_checker(mock_mps, mock_cuda, mock_from_pretrained):
    """The production IllustrationGenerator must disable SD1.5's NSFW safety checker.

    Regression guard for the 2026-10 audit finding (PLAN.md "Discovered"): SD1.5's
    default safety checker false-positives on benign children's-book subjects
    (e.g. "apple", "banana", "cherry") and returns a fully-black image -- verified
    live: a "red apple on a white background" prompt produced 100% dark pixels.
    That would black out every page of a real `pageloop generate` PDF, so the
    checker is disabled in the production generator (the same workaround the
    evidence script uses). This unit test locks that in for the default code path.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    gen = IllustrationGenerator(verify_license=False)
    assert gen.pipeline.safety_checker is None


@patch("pageloom.generator.StableDiffusionPipeline.from_pretrained")
@patch("pageloom.generator.torch.cuda.is_available", return_value=False)
@patch("pageloom.generator.torch.backends.mps.is_available", return_value=False)
def test_from_pretrained_called_with_safety_checker_none(mock_mps, mock_cuda, mock_from_pretrained):
    """The checker must be disabled at load time, not only after load.

    With diffusers >= 0.40 a late ``pipeline.safety_checker = None`` leaves the
    unused checker module resident in VRAM/RAM for the whole run; passing
    ``safety_checker=None`` (and ``feature_extractor=None``) to from_pretrained
    never loads it. Locks in the PLAN.md item: assert the load-time kwargs.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_pipeline.feature_extractor = object()  # simulate a loader that still set it
    mock_from_pretrained.return_value = mock_pipeline
    from pageloom.generator import IllustrationGenerator

    IllustrationGenerator(verify_license=False)
    kwargs = mock_from_pretrained.call_args.kwargs
    assert kwargs.get("safety_checker") is None
    assert kwargs.get("feature_extractor") is None
    # belt-and-braces fallback for overridden loaders still applies
    assert mock_pipeline.safety_checker is None
    assert mock_pipeline.feature_extractor is None


def _make_generator_with_pipeline_side_effect(side_effect):
    """Build a generator whose mocked pipeline applies `side_effect` on call."""
    from pageloom.generator import IllustrationGenerator

    mock_pipeline = MagicMock()
    mock_pipeline.to.side_effect = lambda device: mock_pipeline
    mock_pipeline.side_effect = side_effect
    with (
        patch("pageloom.generator.torch.cuda.is_available", return_value=False),
        patch("pageloom.generator.torch.backends.mps.is_available", return_value=False),
        patch("pageloom.generator.StableDiffusionPipeline.from_pretrained") as mock_fp,
    ):
        mock_fp.return_value = mock_pipeline
        gen = IllustrationGenerator(verify_license=False)
    return gen, mock_pipeline


def test_round_to_multiple_8_rounds_up():
    from pageloom.generator import IllustrationGenerator

    assert IllustrationGenerator._round_to_multiple_8(618) == 624
    assert IllustrationGenerator._round_to_multiple_8(800) == 800
    assert IllustrationGenerator._round_to_multiple_8(612) == 616
    assert IllustrationGenerator._round_to_multiple_8(624) == 624
    assert IllustrationGenerator._round_to_multiple_8(1) == 8
    assert IllustrationGenerator._round_to_multiple_8(0) == 0


def test_generate_rounds_odd_dimensions_before_pipeline():
    """generate() must round odd library-caller dimensions before hitting the pipeline.

    Regression for the PLAN.md Discovered item: only Pipeline.run() rounded, so a
    library user calling generate(width=618) directly would hit diffusers'
    check_inputs() ValueError. generate() now rounds (and catches ValueError as
    a backup), so a successful pipeline call must see multiples of 8.
    """
    from PIL import Image

    called_with = []

    def side_effect(**kwargs):
        called_with.append((kwargs.get("width"), kwargs.get("height")))
        # Real diffusers check_inputs, so an unrounded size fails the test.
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
            StableDiffusionPipeline,
        )

        StableDiffusionPipeline.check_inputs(
            None, kwargs.get("prompt"), kwargs.get("height"), kwargs.get("width"), 1
        )
        return MagicMock(
            images=[Image.new("RGB", (kwargs.get("width"), kwargs.get("height")), color="green")]
        )

    gen, _ = _make_generator_with_pipeline_side_effect(side_effect)
    result = gen.generate("a happy cat", width=618, height=800)
    assert called_with == [(624, 800)]
    assert result.size == (624, 800)


def test_generate_catches_divisible_by_8_value_error_and_retries_once():
    """When the pipeline raises the divisible-by-8 ValueError despite valid input, retry once.

    Covers the PLAN.md Discovered item's "round and retry" wording: generate()
    catches the ValueError, rounds again, and returns a successful image at the
    rounded size instead of propagating the raw exception.
    """
    from PIL import Image

    calls = []

    def side_effect(**kwargs):
        calls.append((kwargs.get("width"), kwargs.get("height")))
        if len(calls) == 1:
            # Simulate a pipeline that rejects even (valid) sizes once, e.g. a
            # stricter or misconfigured check. generate() must catch this,
            # round again, and retry.
            raise ValueError("`height` and `width` have to be divisible by 8 but are 800 and 618.")
        return MagicMock(
            images=[Image.new("RGB", (kwargs.get("width"), kwargs.get("height")), color="blue")]
        )

    gen, _ = _make_generator_with_pipeline_side_effect(side_effect)
    # start with an already-multiple-of-8 size so only the retry path rounds
    result = gen.generate("a happy cat", width=624, height=800)
    assert len(calls) == 2
    assert calls[1] == (632, 808)  # bumped to next multiple of 8
    assert result.size == (632, 808)
    assert result.getpixel((0, 0)) == (0, 0, 255)


def test_generate_propagates_unrelated_value_error():
    """An unrelated ValueError (e.g. prompt/embeds) must not be swallowed."""

    def side_effect(**kwargs):
        raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")

    gen, _ = _make_generator_with_pipeline_side_effect(side_effect)
    try:
        gen.generate("a happy cat", width=512, height=512)
    except ValueError as exc:
        assert "prompt_embeds" in str(exc)
    else:
        raise AssertionError("expected ValueError to propagate")
