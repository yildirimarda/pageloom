"""Plain image-set generation with no page, letter, or text-overlay logic.

Renders one image per prompt (taken from positional arguments or a content
file), optionally conditioned on an optional style reference image. Unlike
:class:`pageloom.pipeline.Pipeline`, this path knows nothing about pages,
letters, words, layout bands, consistency guards, or text overlays: it works
for any image set, not only for pages.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from pageloom.providers import ImageProvider, ImageProviderCapabilities, build_image_provider
from pageloom.providers.size_negotiation import generate_fitting_slot, negotiate_size


def round_to_multiple(dim: int, base: int = 8) -> int:
    """Round ``dim`` up to the nearest multiple of ``base`` (default 8).

    Stable Diffusion's VAE requires generation dimensions to be multiples of 8,
    so caller-supplied sizes are rounded before they reach a real provider.
    """
    if base <= 0:
        raise ValueError("base must be positive")
    return ((int(dim) + base - 1) // base) * base


def load_contents(prompts: list[str], content_file: str | Path | None) -> list[str]:
    """Merge positional prompts with the items in an optional content file.

    A content file may be plain text (one prompt per line), CSV (first column),
    or JSON (a list of strings, or a list of objects with a ``prompt`` key).
    Blank lines and ``#`` comments are skipped.
    """
    contents = [prompt for prompt in prompts if prompt and prompt.strip()]
    if content_file is not None:
        contents.extend(_load_content_file(content_file))
    return contents


def _load_content_file(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Content file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in content file {path} at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(data, list):
            raise ValueError(f"JSON content file {path} must contain a list of prompts")
        items: list[str] = []
        for idx, entry in enumerate(data, start=1):
            if isinstance(entry, str):
                items.append(entry)
            elif isinstance(entry, dict) and "prompt" in entry:
                items.append(str(entry["prompt"]))
            else:
                raise ValueError(f"Unsupported JSON entry at line {idx} in {path}: {entry!r}")
        return [item for item in items if item and item.strip()]
    # Plain text or CSV: one prompt per line.
    lines: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _image_digest(image: Image.Image) -> str:
    """Stable md5 of a PIL image's PNG bytes, used as a cache-key ingredient."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.md5(buffer.getvalue()).hexdigest()


class ImageSetRenderer:
    """Generate one image per prompt, with no page/letter/text-overlay logic.

    An optional style reference image is forwarded to providers that support
    reference images (via ``capabilities.max_reference_images``) so outputs can
    match its style; providers without reference support use the prompt alone.
    """

    def __init__(
        self,
        contents: list[str],
        *,
        provider: ImageProvider | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        style_reference: str | Path | Image.Image | None = None,
        width: int = 512,
        height: int = 512,
        force: bool = False,
        cache_dir: str | Path | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self.contents = [c for c in contents if c and c.strip()]
        self.provider = provider or build_image_provider(provider=provider_name, model=model)
        if isinstance(style_reference, (str, Path)):
            self.style_reference: Image.Image | None = Image.open(str(style_reference)).convert(
                "RGB"
            )
        else:
            self.style_reference = style_reference
        if self.style_reference is not None and not isinstance(self.style_reference, Image.Image):
            raise TypeError("style_reference must be a path or a PIL image")
        self.width = round_to_multiple(max(1, int(width)))
        self.height = round_to_multiple(max(1, int(height)))
        self.force = force
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_cost_usd = max_cost_usd
        self.total_cost_usd = 0.0
        self.budget_stopped = False

    def estimated_total_cost_usd(self) -> float:
        """Estimated USD cost for every image, without calling the provider."""
        capabilities = getattr(self.provider, "capabilities", None)
        gen_width, gen_height = negotiate_size(self.width, self.height, capabilities)
        per_image = self.provider.estimated_cost_usd(gen_width, gen_height)
        return len(self.contents) * per_image

    def _cache_path(self, content: str) -> Path | None:
        if self.cache_dir is None:
            return None
        reference_digest = (
            _image_digest(self.style_reference) if self.style_reference is not None else "-"
        )
        key = hashlib.md5(
            f"{content}|{self.width}|{self.height}|{reference_digest}".encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.png"

    def render(self) -> list[Image.Image]:
        """Generate one image per prompt and return them as PIL images."""
        images: list[Image.Image] = []
        capabilities = getattr(self.provider, "capabilities", None)
        supports_references = (
            isinstance(capabilities, ImageProviderCapabilities)
            and capabilities.max_reference_images > 0
        )
        references = (
            [self.style_reference]
            if (supports_references and self.style_reference is not None)
            else None
        )
        import sys

        for content in self.contents:
            cache_path = self._cache_path(content)
            if not self.force and cache_path is not None and cache_path.exists():
                image = Image.open(str(cache_path))
            else:
                image_cost = self.provider.estimated_cost_usd(self.width, self.height)
                if (
                    self.max_cost_usd is not None
                    and self.total_cost_usd + image_cost > self.max_cost_usd
                ):
                    self.budget_stopped = True
                    break
                image = generate_fitting_slot(
                    self.provider,
                    prompt=content,
                    width=self.width,
                    height=self.height,
                    references=references,
                )
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(str(cache_path), "PNG")
                self.total_cost_usd += image_cost
            images.append(image)

        if self.budget_stopped and self.max_cost_usd is not None:
            print(
                f"Warning: max cost ${self.max_cost_usd:.2f} reached; stopped "
                f"before exceeding the budget (${self.total_cost_usd:.2f} used).",
                file=sys.stderr,
            )
        return images
