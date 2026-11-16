"""OpenAI image provider using the images.generate and images.edit endpoints."""

from __future__ import annotations

import base64
import io
import os

from PIL import Image

from pageloom import settings
from pageloom.providers.base import ImageProviderCapabilities


class OpenAIImageProvider:
    """Generate and edit images via OpenAI's Images API.

    Uses ``images.generate`` for text-to-image generation and ``images.edit``
    when a reference image (the example page) is supplied, so the example page
    conditions the generated artwork. The model id is read from settings
    (``OPENAI_IMAGE_MODEL`` env var, default ``gpt-image-2``).
    """

    provider_name = "openai"

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
        self.model_id = model_id or settings.get_openai_image_model()
        self.api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("PAGELOOM_LLM_API_KEY")
        )
        # No local license check for an online provider; accept the arg for
        # interface conformance with ``build_image_provider``.
        self.verify_license = verify_license

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI image generation requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        return openai.OpenAI(api_key=self.api_key or "sk-dummy")

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> io.BytesIO:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf

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

        When ``references`` is non-empty and the provider supports edits
        (``capabilities.supports_edit``), the first reference (the example
        page) is sent to the ``images.edit`` endpoint. Otherwise
        ``images.generate`` is used.
        """
        del negative_prompt, seed  # not supported by the Images API shape used here
        client = self._client()
        size = f"{max(1, int(width))}x{max(1, int(height))}"
        reference = references[0] if references else None

        if reference is not None and self.capabilities.supports_edit:
            return self._edit(client, prompt, reference, size)
        return self._generate(client, prompt, size)

    def _generate(self, client, prompt: str, size: str) -> Image.Image:
        response = client.images.generate(
            model=self.model_id,
            prompt=prompt,
            n=1,
            size=size,
        )
        return self._image_from_response(response)

    def _edit(self, client, prompt: str, reference, size: str) -> Image.Image:
        image = reference
        if isinstance(image, Image.Image):
            image = self._image_to_bytes(image)
        response = client.images.edit(
            model=self.model_id,
            image=image,
            prompt=prompt,
            n=1,
            size=size,
        )
        return self._image_from_response(response)

    @staticmethod
    def _image_from_response(response) -> Image.Image:
        data = response.data[0]
        b64 = getattr(data, "b64_json", None)
        if b64:
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        url = getattr(data, "url", None)
        if url:
            import urllib.request

            with urllib.request.urlopen(url) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        raise RuntimeError("OpenAI image response contained no image data")


__all__ = ["OpenAIImageProvider"]
