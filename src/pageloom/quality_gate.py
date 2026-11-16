"""Cheap, deterministic image-quality heuristics used by the quality gate.

The quality gate rejects generated frames that are unusable: near-black (the
SD1.5 safety-checker blackout), near-uniform (a blank/white page), tiled (a
small pattern repeated across the frame) or noise-like (unstructured random
snow). These offline heuristics run entirely without a model and are
deterministic, so they are safe to assert against in both offline and live
tests.

The vision-model subject check, retry-on-reject and PDF failure reporting live
in :mod:`pageloom.pipeline`; this module only decides whether a single image
passes or fails the cheap-heuristics half of the gate.
"""

from __future__ import annotations

from PIL import Image

_BLACK_THRESHOLD = 8
_UNIFORM_THRESHOLD = 16
_TILED_THRESHOLD = 10
_NOISE_THRESHOLD = 45


class QualityGateError(RuntimeError):
    """Raised when an image cannot be made to pass the quality gate.

    Carries the list of failing heuristic reasons so callers can report exactly
    why a frame was rejected instead of silently writing it into the PDF.
    """

    def __init__(self, reasons: list[str], image: Image.Image | None = None) -> None:
        self.reasons = list(reasons)
        self.image = image
        super().__init__("image failed quality gate: " + ", ".join(self.reasons))


def is_black(image: Image.Image, threshold: int = _BLACK_THRESHOLD) -> bool:
    """True when every pixel is near-black (SD1.5 safety-checker blackout)."""
    extrema = image.convert("RGB").getextrema()
    return all(channel_max <= threshold for _lo, channel_max in extrema)


def is_uniform(image: Image.Image, threshold: int = _UNIFORM_THRESHOLD) -> bool:
    """True when the image is near-uniform (a single dominant color)."""
    extrema = image.convert("RGB").getextrema()
    return all(channel_max - channel_min <= threshold for channel_min, channel_max in extrema)


def is_tiled(image: Image.Image, threshold: int = _TILED_THRESHOLD) -> bool:
    """True when the image looks like a small tile repeated across the frame.

    A generation that collapsed onto a repeated noise tile shows the same few
    block values in every quadrant. The image is downscaled to a small grid,
    split into 4x4 blocks, and treated as tiled when the block means repeat
    with a small period (1 or 2). Uniform images are already caught by
    :func:`is_uniform`, so this targets multi-colour repetition rather than a
    blank page.
    """
    if is_uniform(image):
        return False
    gray = image.convert("L").resize((16, 16))
    pixels = gray.tobytes()

    def block_mean(by: int, bx: int) -> float:
        total = 0
        for dy in range(4):
            for dx in range(4):
                total += pixels[(by * 4 + dy) * 16 + bx * 4 + dx]
        return total / 16.0

    for period in (1, 2):
        repeated = True
        for by in range(4):
            for bx in range(4):
                if abs(block_mean(by, bx) - block_mean(by % period, bx % period)) > threshold:
                    repeated = False
                    break
            if not repeated:
                break
        if repeated:
            return True
    return False


def is_noise_like(image: Image.Image, threshold: int = _NOISE_THRESHOLD) -> bool:
    """True when the image looks like unstructured random noise.

    A noise frame has high variation between adjacent pixels in both
    directions but no large-scale structure. Black and uniform frames are
    already caught by :func:`is_black`/:func:`is_uniform`, so this targets the
    high-frequency "snow" a bad generation produces.
    """
    gray = image.convert("L")
    width, height = gray.size
    if width < 2 or height < 2:
        return False
    pixels = gray.tobytes()
    total_h = total_v = 0
    count_h = count_v = 0
    for y in range(height):
        base = y * width
        for x in range(width - 1):
            total_h += abs(pixels[base + x] - pixels[base + x + 1])
            count_h += 1
    for y in range(height - 1):
        base = y * width
        for x in range(width):
            total_v += abs(pixels[base + x] - pixels[base + width + x])
            count_v += 1
    mean_h = total_h / count_h if count_h else 0.0
    mean_v = total_v / count_v if count_v else 0.0
    return mean_h > threshold and mean_v > threshold


def quality_gate_reasons(image: Image.Image) -> list[str]:
    """Return a list of reasons the image fails the gate, or [] when it passes."""
    reasons: list[str] = []
    if is_black(image):
        reasons.append("near-black frame")
    if is_uniform(image):
        reasons.append("near-uniform frame")
    if is_tiled(image):
        reasons.append("tiled frame")
    if is_noise_like(image):
        reasons.append("noise-like frame")
    return reasons


def assert_passes_quality_gate(image: Image.Image) -> None:
    """Assert the image passes the quality gate.

    Raises :class:`AssertionError` naming each failed heuristic.
    """
    reasons = quality_gate_reasons(image)
    if reasons:
        raise AssertionError("image failed quality gate: " + ", ".join(reasons))
