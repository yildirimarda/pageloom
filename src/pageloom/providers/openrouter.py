"""OpenRouter image provider using chat completions with image output modality.

Generates images via OpenRouter's chat completions endpoint with the
``modalities`` option set to ``["image"]``, so the selected model returns an
image instead of text. The model id is read from settings
(``OPENROUTER_IMAGE_MODEL`` env var).

A helper, :func:`list_openrouter_image_models`, fetches OpenRouter's models
endpoint and returns model ids that support image output, so users can pick a
current image-output model.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

from PIL import Image

from pageloom import settings
from pageloom.providers.base import ImageProviderCapabilities

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterImageProvider:
    """Generate images via OpenRouter chat completions with image output modality.

    Sends a chat completion with ``modalities: ["image"]`` so the model returns
    generated image data. Reference images (the example page) are accepted for
    interface conformance but not forwarded, since the chat-completions image
    path used here is text-to-image only (see ``capabilities``).
    """

    provider_name = "openrouter"

    capabilities = ImageProviderCapabilities(
        max_reference_images=1,
        supported_sizes=((1024, 1024),),
        supported_aspect_ratios=(),
        supports_negative_prompt=False,
        supports_edit=False,
    )

    def __init__(
        self,
        model_id: str | None = None,
        verify_license: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id or settings.get_openrouter_image_model()
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("PAGELOOM_LLM_API_KEY")
        )
        # No local license check for an online provider; accept the arg for
        # interface conformance with ``build_image_provider``.
        self.verify_license = verify_license

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenRouter image generation requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        return openai.OpenAI(
            api_key=self.api_key or "sk-dummy",
            base_url=_OPENROUTER_BASE_URL,
        )

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        references: list[str | Image.Image] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate one image for ``prompt`` via OpenRouter chat completions.

        ``modalities`` is set to ``["image"]`` so the model returns image
        output. ``negative_prompt`` and ``seed`` are not part of the request
        shape used here and are accepted only for interface conformance.
        """
        del negative_prompt, seed, references  # not supported by this request shape
        client = self._client()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "modalities": ["image"],
                "width": max(1, int(width)),
                "height": max(1, int(height)),
            },
        )
        return self._image_from_response(response)

    @staticmethod
    def _image_from_response(response) -> Image.Image:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenRouter image response contained no choices")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("OpenRouter image response contained no message")
        images = getattr(message, "images", None) or []
        for image in images:
            b64 = getattr(image, "b64_json", None)
            if b64:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            image_url = getattr(image, "image_url", None)
            url = getattr(image_url, "url", None) if image_url is not None else None
            if url:
                with urllib.request.urlopen(url) as resp:
                    return Image.open(io.BytesIO(resp.read())).convert("RGB")
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.startswith("data:image"):
            _, b64 = content.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        raise RuntimeError("OpenRouter image response contained no image data")


def list_openrouter_image_models(api_key: str | None = None) -> list[str]:
    """Return OpenRouter model ids that support image output.

    Fetches https://openrouter.ai/api/v1/models and keeps models whose
    architecture modality indicates image output (e.g. ``text->image``,
    ``image->image``), so users can pick a current image-output model id to
    set via ``OPENROUTER_IMAGE_MODEL`` or ``--image-model``.
    """
    api_key = (
        api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("PAGELOOM_LLM_API_KEY")
    )
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{_OPENROUTER_BASE_URL}/models", headers=headers, method="GET"
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    image_models: list[str] = []
    for model in data.get("data", []):
        architecture = model.get("architecture") or {}
        modality = str(architecture.get("modality", ""))
        parts = [part.strip() for part in modality.split("->")]
        if len(parts) == 2 and parts[1] == "image":
            model_id = model.get("id")
            if model_id:
                image_models.append(model_id)
    return image_models


__all__ = ["OpenRouterImageProvider", "list_openrouter_image_models"]
