"""Stable Diffusion illustration generator using Hugging Face diffusers."""

from __future__ import annotations

import sys

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from pageloom import settings
from pageloom.license_check import LicenseVerifier

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


class IllustrationGenerator:
    """Generate illustrations using a Stable Diffusion pipeline."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        verify_license: bool = True,
        license_context: str = "book intended for sale",
    ) -> None:
        self.model_id = model_id if model_id is not None else settings.get_model_id()
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
        self.torch_dtype = torch_dtype or torch.float32
        if verify_license:
            verifier = LicenseVerifier(model_id=self.model_id)
            verifier.verify(
                intended_use="commercial",
                context=license_context,
            )
        cache_dir = settings.get_hf_cache_dir()
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Stable Diffusion model '{self.model_id}': {exc}"
            ) from exc
        self.pipeline = self.pipeline.to(self.device)
        # Attention slicing enabled consistently for lower memory usage
        self.pipeline.enable_attention_slicing()
        # SD1.5's bundled safety checker false-positives on benign children's-book
        # subjects (e.g. "banana", "cherry", "apple") and blacks out the whole
        # image, silently destroying the illustration we were asked to draw --
        # verified live (2026-10 audit: a "red apple on a white background" prompt
        # returned 100% black pixels). PageLoom only ever generates family-friendly
        # content (letters, numbers, fruit, animals), so disable the safety filter
        # to keep generated pages visually sensible. This mirrors the workaround
        # already used by scripts/generate_real_multiple_labels.py; making it the
        # production default is the recorded Discovered gap (see PLAN.md).
        self.pipeline.safety_checker = None
        # Belt and braces: disabling the checker above is the primary fix, but
        # generate() also black-frame-detects and retries (with a fresh seed)
        # in case a caller re-enables the checker or a future model ships a
        # different blackout path -- a black frame is never a valid result.
        self.safety_max_retries = _safety_max_retries()

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
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
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
            try:
                result = self.pipeline(
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
