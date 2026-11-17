"""Stable Diffusion illustration generator backed by Hugging Face diffusers.

Implements the :class:`~pageloom.providers.base.ImageProvider` interface so the
rest of the pipeline (notably :class:`~pageloom.pipeline.Pipeline`) depends only
on the interface, never on a concrete Stable Diffusion class.
"""

from __future__ import annotations

import sys

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

from pageloom import settings
from pageloom.license_check import LicenseVerifier
from pageloom.providers.base import ImageProviderCapabilities

# Abort the safety-checker retry loop after this many consecutive
# NSFW-flagged (black) results. Set PAGELOOM_SAFETY_MAX_RETRIES=0 to disable
# retries entirely (still tolerant: the black image is returned as-is, as
# before), or a higher value to retry more aggressively.
DEFAULT_SAFETY_MAX_RETRIES = 3


def _safety_max_retries() -> int:
    raw = __import__("os").environ.get("PAGELOOM_SAFETY_MAX_RETRIES")
    if raw is None or raw.strip() == "":
        return DEFAULT_SAFETY_MAX_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_SAFETY_MAX_RETRIES


def _is_black_image(image: Image.Image, threshold: int = 8) -> bool:
    """True when every pixel is near-black (the NSFW safety checker's blackout).

    SD1.5's bundled ``StableDiffusionSafetyChecker`` replaces a flagged image
    with ``np.zeros(...)`` (fully black). A genuinely dark illustration can
    contain a few non-zero pixels, so require every channel to be within
    ``threshold`` of 0 before calling it a blackout.
    """
    extrema = image.convert("RGB").getextrema()
    return all(channel_max <= threshold for _lo, channel_max in extrema)


def _native_generation_size(model_id: str) -> int:
    """Return the native square generation size for a Stable Diffusion model family.

    SD 1.5 was trained at 512 x 512 and SDXL-class models at 1024 x 1024.
    Generating at any other size -- page size or narrow strips -- produces
    duplicated, tiled noise because the model was never trained there; see
    docs/IMAGE-BACKENDS.md. The model id is configuration, so the family is
    inferred from it rather than hardcoding a size per provider.
    """
    lowered = model_id.lower()
    if "sdxl" in lowered or "stable-diffusion-xl" in lowered:
        return 1024
    return 512


class LocalDiffusersProvider:
    """Generate illustrations using a Stable Diffusion pipeline.

    A concrete :class:`~pageloom.providers.base.ImageProvider` for local Stable
    Diffusion via Hugging Face ``diffusers``.
    """

    provider_name = "local"

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        verify_license: bool = True,
        license_context: str = "book intended for sale",
        image_strength: float | None = None,
    ) -> None:
        self.model_id = model_id if model_id is not None else settings.get_model_id()
        self.native_size = _native_generation_size(self.model_id)
        # The local provider can take one reference image (the example page)
        # for image-to-image conditioning; whether it actually does so is
        # controlled by ``image_strength`` (off by default).
        self.capabilities = ImageProviderCapabilities(
            supported_sizes=((self.native_size, self.native_size),),
            supports_negative_prompt=True,
            max_reference_images=1,
        )
        self.verify_license = verify_license
        self.license_context = license_context
        if device is None:
            env_device = settings.get_device()
            if env_device is not None:
                device = env_device
            elif torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        elif device in ("cuda", "mps"):
            # float16 halves VRAM and is the production dtype on the GPU
            # backends; CPU stays on float32 (float16 is unsupported/slow there).
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        self.num_inference_steps = settings.get_num_inference_steps()
        self.guidance_scale = settings.get_guidance_scale()
        # Image-to-image strength: 0.0 (default) means plain text-to-image,
        # the example page is never used as a reference. Any value in
        # (0, 1] enables image-to-image conditioning on the example page.
        self.image_strength = (
            settings.get_image_strength() if image_strength is None else image_strength
        )
        if verify_license:
            verifier = LicenseVerifier(model_id=self.model_id)
            verifier.verify(
                intended_use="commercial",
                context=license_context,
            )
        cache_dir = settings.get_hf_cache_dir()
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "cache_dir": cache_dir,
            "safety_checker": None,
            "feature_extractor": None,
        }
        try:
            self.pipeline = self._load_pipeline(StableDiffusionPipeline, **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download/load Stable Diffusion model '{self.model_id}'. "
                f"Check the model id (overridable with PAGELOOM_MODEL_ID or "
                f"--image-model) and your network connection. "
                f"Underlying error: {exc}"
            ) from exc
        self.pipeline = self.pipeline.to(self.device)
        self._finalize_pipeline(self.pipeline)
        # The image-to-image pipeline is only loaded when reference
        # conditioning is enabled (strength > 0); otherwise it stays None and
        # generate() always uses the text-to-image pipeline above.
        self.img2img_pipeline: StableDiffusionImg2ImgPipeline | None = None
        if self.image_strength > 0:
            try:
                self.img2img_pipeline = self._load_pipeline(
                    StableDiffusionImg2ImgPipeline, **load_kwargs
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download/load Stable Diffusion image-to-image "
                    f"model '{self.model_id}'. "
                    f"Underlying error: {exc}"
                ) from exc
            self.img2img_pipeline = self.img2img_pipeline.to(self.device)
            self._finalize_pipeline(self.img2img_pipeline)
        # Belt and braces: disabling the checker above is the primary fix, but
        # generate() also black-frame-detects and retries (with a fresh seed)
        # in case a caller re-enables the checker or a future model ships a
        # different blackout path -- a black frame is never a valid result.
        self.safety_max_retries = _safety_max_retries()

    def _load_pipeline(self, pipeline_cls, **kwargs):
        """Load a Stable Diffusion pipeline (txt2img or img2img) from pretrained.

        Wraps ``from_pretrained`` so the same kwargs (dtype, cache dir, disabled
        safety checker) apply to both pipeline families.
        """
        return pipeline_cls.from_pretrained(self.model_id, **kwargs)

    def _finalize_pipeline(self, pipeline) -> None:
        """Apply the production configuration common to both pipeline families.

        Attention slicing for memory, the DPM++ multistep scheduler for quality
        at production step counts, and the disabled safety checker. Mirrors the
        previous post-load setup, factored out so both ``self.pipeline`` and
        ``self.img2img_pipeline`` get it.
        """
        # Attention slicing enabled consistently for lower memory usage.
        pipeline.enable_attention_slicing()
        # Production scheduler: DPM++ multistep (DPMSolverMultistepScheduler)
        # gives good image quality at the 25-30 production inference steps.
        # Only swap in a real scheduler when the pipeline exposed a config dict
        # (a mocked pipeline in tests has a MagicMock config, which from_config
        # would otherwise mistake for a model path and try to download).
        scheduler_config = getattr(pipeline.scheduler, "config", None)
        if isinstance(scheduler_config, dict):
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
        # SD1.5's bundled safety checker false-positives on benign
        # subjects (e.g. "banana", "cherry", "apple") and
        # blacks out the whole image. Disabled at load time via
        # ``safety_checker=None``/``feature_extractor=None``; the late
        # assignment below remains as a belt-and-braces fallback.
        pipeline.safety_checker = None
        if getattr(pipeline, "feature_extractor", None) is not None:
            pipeline.feature_extractor = None

    def estimated_cost_usd(self, width: int, height: int) -> float:
        """Local Stable Diffusion runs on the user's own hardware: no per-image fee."""
        return 0.0

    @staticmethod
    def _round_to_multiple_8(dim: int) -> int:
        """Round a generation dimension up to the nearest multiple of 8.

        Stable Diffusion's VAE requires both dimensions to be multiples of 8,
        or ``check_inputs()`` raises ``ValueError``. Example-page dimensions
        come from the source image's raw pixel size and are not guaranteed to
        satisfy that constraint, so this rounds up by at most 7 pixels.
        """
        return ((int(dim) + 7) // 8) * 8

    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        references: list[str | Image.Image] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> Image.Image:
        """Generate one illustration for ``prompt``.

        When ``image_strength`` is greater than 0 (off by default) and at least
        one reference is supplied, the example page (``references[0]``) is used
        as the reference for image-to-image generation with
        ``strength=self.image_strength``; otherwise plain text-to-image is used
        and ``references`` is ignored.

        ``num_inference_steps`` and ``guidance_scale`` fall back to the
        production defaults stored on the instance (25 steps, ~7 guidance) when
        not supplied explicitly.
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        max_retries = getattr(self, "safety_max_retries", _safety_max_retries())
        attempt = 0
        # Stable Diffusion's VAE requires width/height to be multiples of 8;
        # `Pipeline.run()` rounds before calling us, but a library caller can
        # pass odd sizes directly. Defense-in-depth: round up and retry once
        # when the pipeline raises `check_inputs()`' ValueError.
        width = self._round_to_multiple_8(width)
        height = self._round_to_multiple_8(height)
        rounded = False
        while True:
            # Vary the latent seed per attempt so a safety-checker blackout
            # (which is seed-dependent) gets a fresh draw rather than
            # reproducing the same black image forever.
            attempt_seed = None if seed is None else seed + attempt
            generator = None
            if attempt_seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(attempt_seed)
            use_img2img = (
                self.image_strength > 0 and self.img2img_pipeline is not None and references
            )
            pipeline = self.img2img_pipeline if use_img2img else self.pipeline
            try:
                if use_img2img:
                    # Image-to-image: the example page is the reference and
                    # ``strength`` controls how closely the output follows it.
                    result = pipeline(
                        prompt=prompt,
                        image=references[0],
                        strength=self.image_strength,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
                else:
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
            except ValueError as exc:
                message = str(exc)
                if (
                    not rounded
                    and "divisible by 8" in message
                    and width % 8 == 0
                    and height % 8 == 0
                ):
                    # Already rounded above, yet still rejected: bump to the
                    # next multiple of 8 in case the failure reported different
                    # dimensions than we passed (defensive, retry once only).
                    rounded = True
                    width = self._round_to_multiple_8(width + 8)
                    height = self._round_to_multiple_8(height + 8)
                    print(
                        "Warning: pipeline rejected width/height as not divisible "
                        f"by 8 (got {width}x{height}); retrying rounded to "
                        f"{width}x{height}.",
                        file=sys.stderr,
                    )
                    continue
                raise
            images = getattr(result, "images", None) or []
            if not images:
                return Image.new("RGB", (width, height), color="gray")
            image = images[0]
            if not isinstance(image, Image.Image):
                return Image.new("RGB", (width, height), color="gray")
            # Tolerate SD1.5's false-positive NSFW blackout: retry with a new
            # seed instead of silently returning the black frame. If the
            # configured pipeline still flags us after ``max_retries``, return
            # the black image rather than looping forever (same end state as
            # before this retry existed, just with a clear stderr warning).
            if _is_black_image(image) and attempt < max_retries:
                attempt += 1
                print(
                    "Warning: SD1.5 safety checker blacked out an illustration "
                    f"(NSFW false positive on a benign prompt); retrying "
                    f"({attempt}/{max_retries}) with a different seed.",
                    file=sys.stderr,
                )
                continue
            if _is_black_image(image):
                print(
                    "Warning: SD1.5 safety checker blacked out an illustration "
                    f"after {attempt + 1} attempt(s); returning the black image. "
                    "Set PAGELOOM_SAFETY_MAX_RETRIES to raise the retry cap.",
                    file=sys.stderr,
                )
            return image
