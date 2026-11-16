"""Google Gemini image provider for the Nano Banana family.

Generates images via Google's ``google.genai`` SDK, calling
``models.generate_content`` with the example page supplied as a reference
image (an image part) when ``references`` are given. The model id is read
from settings (``GEMINI_IMAGE_MODEL`` env var, default
``gemini-nano-banana-2.1``).
"""

from __future__ import annotations

import io
import os

from PIL import Image

from pageloom import settings
from pageloom.providers.base import ImageProviderCapabilities


class GeminiImageProvider:
    """Generate images via Google's Gemini Nano Banana family.

    Uses ``google.genai.Client.models.generate_content`` with the prompt plus
    any reference images passed as content parts. When a reference (the
    example page) is supplied it is included in the request so the generated
    artwork conditions on it; otherwise a plain text-to-image call is made.
    """

    provider_name = "gemini"

    capabilities = ImageProviderCapabilities(
        max_reference_images=1,
        supported_sizes=((1024, 1024),),
        supported_aspect_ratios=(),
        supports_negative_prompt=False,
        supports_edit=True,
    )

    def __init__(
        self,
        model_id: str | None = None,
        verify_license: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id or settings.get_gemini_image_model()
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("PAGELOOM_LLM_API_KEY")
        )
        # No local license check for an online provider; accept the arg for
        # interface conformance with ``build_image_provider``.
        self.verify_license = verify_license

    def _client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini image generation requires the 'google-genai' package. "
                "Install it with: pip install google-genai"
            ) from exc
        return genai.Client(api_key=self.api_key or "dummy")

    @staticmethod
    def _prepare_references(
        references: list[str | Image.Image] | None,
    ) -> list[Image.Image]:
        """Load file-path references as PIL images; pass images straight through."""
        prepared: list[Image.Image] = []
        for reference in references or []:
            if isinstance(reference, Image.Image):
                prepared.append(reference)
            else:
                prepared.append(Image.open(reference).convert("RGB"))
        return prepared

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        references: list[str | Image.Image] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate one image for ``prompt``.

        Reference images (the example page) are appended to the request
        contents as image parts so the Nano Banana family can condition on
        them. ``negative_prompt`` and ``seed`` are not part of the request
        shape used here and are accepted only for interface conformance.
        """
        del negative_prompt, seed
        client = self._client()
        contents: list = [prompt]
        contents.extend(self._prepare_references(references))
        response = client.models.generate_content(
            model=self.model_id,
            contents=contents,
        )
        return self._image_from_response(response)

    @staticmethod
    def _image_from_response(response) -> Image.Image:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise RuntimeError("Gemini image response contained no candidates")
        parts = getattr(candidates[0].content, "parts", None) or []
        if not parts:
            raise RuntimeError("Gemini image response contained no parts")
        inline_data = getattr(parts[0], "inline_data", None)
        data = getattr(inline_data, "data", None) if inline_data is not None else None
        if data:
            return Image.open(io.BytesIO(data)).convert("RGB")
        raise RuntimeError("Gemini image response contained no image data")

    def estimated_cost_usd(self, width: int, height: int) -> float:
        """Estimated USD cost for one generated image from the price table."""
        return settings.image_price_usd("gemini", self.model_id)


__all__ = ["GeminiImageProvider"]
