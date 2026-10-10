"""Stable Diffusion illustration generator using Hugging Face diffusers."""

from __future__ import annotations

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from pageloom import settings
from pageloom.license_check import LicenseVerifier


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
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        images = getattr(result, "images", None) or []
        if not images:
            return Image.new("RGB", (width, height), color="gray")
        image = images[0]
        if not isinstance(image, Image.Image):
            return Image.new("RGB", (width, height), color="gray")
        return image
