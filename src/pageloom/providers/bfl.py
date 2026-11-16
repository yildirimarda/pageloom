"""Black Forest Labs FLUX.2 image provider.

Reachable through Black Forest Labs' own API (``api.bfl.ai``) and, when
``PAGELOOM_FLUX_HOST`` is set to ``fal`` or ``replicate``, through fal.ai or
Replicate respectively. Each host uses its own request shape and credential
environment variable. Supports multi-reference input: the example page plus
up to two additional conditioning images.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image

from pageloom import settings
from pageloom.providers.base import ImageProviderCapabilities

_BFL_BASE = "https://api.bfl.ai/v1"
_FAL_BASE = "https://fal.run/fal-ai"
_REPLICATE_BASE = "https://api.replicate.com/v1"

_POLL_INTERVAL = 2.0
_MAX_POLL_ATTEMPTS = 30


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class BFLImageProvider:
    """Generate images via Black Forest Labs FLUX.2.

    Default host is BFL's own API. Set ``PAGELOOM_FLUX_HOST=fal`` to route
    through fal.ai, or ``PAGELOOM_FLUX_HOST=replicate`` to route through
    Replicate. Each host uses its own request shape and credential env var.
    """

    provider_name = "flux"

    capabilities = ImageProviderCapabilities(
        max_reference_images=3,
        supported_sizes=((1024, 1024),),
        supported_aspect_ratios=((1, 1), (16, 9), (9, 16), (3, 2), (2, 3)),
        supports_negative_prompt=False,
        supports_edit=True,
    )

    def __init__(
        self,
        model_id: str | None = None,
        verify_license: bool = False,
        api_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self.model_id = model_id or settings.get_flux_model()
        self.host = host or settings.get_flux_host()
        self.api_key = api_key or self._default_api_key()
        # No local license check for an online provider; accept the arg for
        # interface conformance with ``build_image_provider``.
        self.verify_license = verify_license

    def _default_api_key(self) -> str | None:
        if self.host == "fal":
            return os.environ.get("FAL_KEY") or os.environ.get("PAGELOOM_FAL_KEY")
        if self.host == "replicate":
            return os.environ.get("REPLICATE_API_KEY") or os.environ.get(
                "PAGELOOM_REPLICATE_API_KEY"
            )
        # bfl
        return os.environ.get("BFL_API_KEY") or os.environ.get("PAGELOOM_BFL_API_KEY")

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        references: list[str | Image.Image] | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate one image for ``prompt`` via the selected FLUX host."""
        del negative_prompt  # not part of any FLUX request shape used here
        references = references or []
        if len(references) > self.capabilities.max_reference_images:
            raise ValueError(
                "FLUX provider accepts at most "
                f"{self.capabilities.max_reference_images} reference images, "
                f"got {len(references)}."
            )

        if self.host == "fal":
            return self._generate_fal(prompt, width, height, references, seed)
        if self.host == "replicate":
            return self._generate_replicate(prompt, width, height, references, seed)
        return self._generate_bfl(prompt, width, height, references, seed)

    # -- Black Forest Labs' own API ---------------------------------------

    def _generate_bfl(self, prompt, width, height, references, seed):
        body: dict = {
            "prompt": prompt,
            "width": max(1, int(width)),
            "height": max(1, int(height)),
            "output_format": "png",
        }
        if seed is not None:
            body["seed"] = seed
        if references:
            body["image_prompt"] = [
                {"image": self._ref_to_url(ref), "strength": 0.8} for ref in references
            ]
        request = urllib.request.Request(
            f"{_BFL_BASE}/{self.model_id}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            launch = json.loads(resp.read().decode("utf-8"))
        request_id = launch.get("id")
        if not request_id:
            raise RuntimeError("BFL FLUX launch returned no request id")
        return self._poll_bfl(request_id)

    def _poll_bfl(self, request_id: str) -> Image.Image:
        for _ in range(_MAX_POLL_ATTEMPTS):
            time.sleep(_POLL_INTERVAL)
            request = urllib.request.Request(
                f"{_BFL_BASE}/get_result?id={request_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            status = result.get("status")
            if status == "Ready":
                images = (result.get("result") or {}).get("sample") or []
                if images:
                    return self._download_image(images[0])
                raise RuntimeError("BFL FLUX result contained no image")
            if status in ("Error", "Failed"):
                raise RuntimeError(f"BFL FLUX generation failed: {result}")
        raise RuntimeError("BFL FLUX generation timed out while polling")

    # -- fal --------------------------------------------------------------

    def _generate_fal(self, prompt, width, height, references, seed):
        body: dict = {
            "prompt": prompt,
            "image_urls": [self._ref_to_url(ref) for ref in references],
            "aspect_ratio": self._aspect_ratio(width, height),
        }
        if seed is not None:
            body["seed"] = seed
        request = urllib.request.Request(
            f"{_FAL_BASE}/{self.model_id}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        image = result.get("image") or {}
        url = image.get("url")
        if not url:
            raise RuntimeError("fal FLUX response contained no image url")
        return self._download_image(url)

    # -- Replicate --------------------------------------------------------

    def _generate_replicate(self, prompt, width, height, references, seed):
        body: dict = {
            "input": {
                "prompt": prompt,
                "image": [self._ref_to_url(ref) for ref in references],
                "aspect_ratio": self._aspect_ratio(width, height),
            }
        }
        if seed is not None:
            body["input"]["seed"] = seed
        request = urllib.request.Request(
            f"{_REPLICATE_BASE}/models/black-forest-labs/{self.model_id}/predictions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            launch = json.loads(resp.read().decode("utf-8"))
        prediction_id = launch.get("id")
        if not prediction_id:
            raise RuntimeError("Replicate FLUX launch returned no prediction id")
        return self._poll_replicate(prediction_id)

    def _poll_replicate(self, prediction_id: str) -> Image.Image:
        for _ in range(_MAX_POLL_ATTEMPTS):
            time.sleep(_POLL_INTERVAL)
            request = urllib.request.Request(
                f"{_REPLICATE_BASE}/predictions/{prediction_id}",
                headers={"Authorization": f"Token {self.api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            status = result.get("status")
            if status == "succeeded":
                output = result.get("output")
                if isinstance(output, list) and output:
                    return self._download_image(output[0])
                if isinstance(output, str):
                    return self._download_image(output)
                raise RuntimeError("Replicate FLUX result contained no image")
            if status in ("failed", "canceled"):
                raise RuntimeError(f"Replicate FLUX generation failed: {result}")
        raise RuntimeError("Replicate FLUX generation timed out while polling")

    # -- shared helpers ---------------------------------------------------

    def _ref_to_url(self, reference: str | Image.Image) -> str:
        if isinstance(reference, Image.Image):
            return _image_to_data_url(reference)
        if isinstance(reference, str) and reference.startswith(("http://", "https://", "data:")):
            return reference
        # Treat as a local file path: read and inline it as a data URL.
        with open(reference, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _aspect_ratio(width: int, height: int) -> str:
        return f"{max(1, int(width))}:{max(1, int(height))}"

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")


__all__ = ["BFLImageProvider"]
