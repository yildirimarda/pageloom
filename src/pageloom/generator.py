"""Stable Diffusion illustration generator using Hugging Face diffusers."""

from __future__ import annotations

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class IllustrationGenerator:
    """Generate illustrations using a Stable Diffusion pipeline."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.torch_dtype = torch_dtype or (torch.float16 if device == "cuda" else torch.float32)
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
        )
        self.pipeline = self.pipeline.to(self.device)
        # Enable attention slicing for lower memory usage
        if device == "cpu":
            self.pipeline.enable_attention_slicing()

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
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        return image
