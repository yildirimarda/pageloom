"""Size negotiation between a layout slot and a provider's native capabilities.

Maps a requested page or sub-image size to the nearest size or aspect ratio the
selected provider supports, generates there, then crops or pads the result back
to the layout slot -- so callers never need to know which sizes a provider
actually accepts.
"""

from __future__ import annotations

from PIL import Image

from pageloom.providers.base import ImageProvider, ImageProviderCapabilities


def negotiate_size(
    requested_width: int,
    requested_height: int,
    capabilities: ImageProviderCapabilities | None,
) -> tuple[int, int]:
    """Map ``(requested_width, requested_height)`` to a native provider size.

    Precedence:
    1. ``capabilities.supported_sizes`` -- nearest supported size by area, ties
       broken by aspect-ratio closeness.
    2. ``capabilities.supported_aspect_ratios`` -- nearest ratio, scaled to the
       largest size that fits within the request while keeping that ratio.
    3. No usable capabilities -- the provider accepts any size; return the
       request unchanged.
    """
    requested_width = max(1, int(requested_width))
    requested_height = max(1, int(requested_height))

    if not isinstance(capabilities, ImageProviderCapabilities):
        return requested_width, requested_height
    if capabilities.supported_sizes:
        return _nearest_supported_size(
            requested_width, requested_height, capabilities.supported_sizes
        )
    if capabilities.supported_aspect_ratios:
        return _size_for_nearest_ratio(
            requested_width, requested_height, capabilities.supported_aspect_ratios
        )
    return requested_width, requested_height


def _aspect_ratio(width: int, height: int) -> float:
    return width / height


def _nearest_supported_size(
    requested_width: int,
    requested_height: int,
    sizes: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    requested_area = requested_width * requested_height
    requested_ratio = _aspect_ratio(requested_width, requested_height)
    best: tuple[int, int] | None = None
    best_key: tuple[int, float] | None = None
    for width, height in sizes:
        area_diff = abs(width * height - requested_area)
        ratio_diff = abs(_aspect_ratio(width, height) - requested_ratio)
        key = (area_diff, ratio_diff)
        if best_key is None or key < best_key:
            best_key = key
            best = (width, height)
    assert best is not None
    return best


def _size_for_nearest_ratio(
    requested_width: int,
    requested_height: int,
    ratios: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    requested_ratio = _aspect_ratio(requested_width, requested_height)
    ratio_w, ratio_h = min(ratios, key=lambda r: abs(_aspect_ratio(r[0], r[1]) - requested_ratio))
    scale = min(requested_width / ratio_w, requested_height / ratio_h)
    return max(1, round(ratio_w * scale)), max(1, round(ratio_h * scale))


def fit_to_slot(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Return ``image`` cropped or padded to exactly ``(target_width, target_height)``.

    The image is centered: excess is cropped from the middle, shortfalls are
    padded with white. The image is never resampled, so generated detail is
    preserved. Non-PIL values (e.g. test doubles) are returned unchanged.
    """
    if not isinstance(image, Image.Image):
        return image
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    src_width, src_height = image.size
    if (src_width, src_height) == (target_width, target_height):
        return image
    left = max(0, (src_width - target_width) // 2)
    top = max(0, (src_height - target_height) // 2)
    right = min(src_width, left + target_width)
    bottom = min(src_height, top + target_height)
    cropped = image.crop((left, top, right, bottom))
    if cropped.size == (target_width, target_height):
        return cropped
    canvas = Image.new(image.mode, (target_width, target_height), color="white")
    paste_x = (target_width - cropped.width) // 2
    paste_y = (target_height - cropped.height) // 2
    canvas.paste(cropped, (paste_x, paste_y))
    return canvas


def generate_fitting_slot(
    provider: ImageProvider,
    prompt: str,
    width: int,
    height: int,
    *,
    references: list[str | Image.Image] | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
) -> Image.Image:
    """Negotiate a native generation size, generate, then fit to the layout slot.

    The returned image is always exactly ``(width, height)``: the size the
    pipeline's layout slot expects, regardless of what the provider natively
    supports.
    """
    gen_width, gen_height = negotiate_size(width, height, getattr(provider, "capabilities", None))
    call_kwargs: dict = {"prompt": prompt, "width": gen_width, "height": gen_height}
    if references is not None:
        call_kwargs["references"] = references
    if negative_prompt is not None:
        call_kwargs["negative_prompt"] = negative_prompt
    if seed is not None:
        call_kwargs["seed"] = seed
    image = provider.generate(**call_kwargs)
    return fit_to_slot(image, width, height)
