"""Cheap, deterministic image-quality heuristics used by the quality gate.

The quality gate rejects generated frames that are unusable: near-black (the
SD1.5 safety-checker blackout) or near-uniform (a blank/white page). These are
the offline "cheap heuristics" half of the quality gate planned for Milestone
10; the vision-model subject check, retry-on-reject and PDF failure reporting
belong to that later milestone. Everything here runs entirely offline and is
deterministic, so it is safe to assert against in both offline and live tests.
"""

from __future__ import annotations

from PIL import Image

_BLACK_THRESHOLD = 8
_UNIFORM_THRESHOLD = 16


def is_black(image: Image.Image, threshold: int = _BLACK_THRESHOLD) -> bool:
    """True when every pixel is near-black (SD1.5 safety-checker blackout)."""
    extrema = image.convert("RGB").getextrema()
    return all(channel_max <= threshold for _lo, channel_max in extrema)


def is_uniform(image: Image.Image, threshold: int = _UNIFORM_THRESHOLD) -> bool:
    """True when the image is near-uniform (a single dominant color)."""
    extrema = image.convert("RGB").getextrema()
    return all(channel_max - channel_min <= threshold for channel_min, channel_max in extrema)


def quality_gate_reasons(image: Image.Image) -> list[str]:
    """Return a list of reasons the image fails the gate, or [] when it passes."""
    reasons: list[str] = []
    if is_black(image):
        reasons.append("near-black frame")
    if is_uniform(image):
        reasons.append("near-uniform frame")
    return reasons


def assert_passes_quality_gate(image: Image.Image) -> None:
    """Assert the image passes the quality gate.

    Raises :class:`AssertionError` naming each failed heuristic.
    """
    reasons = quality_gate_reasons(image)
    if reasons:
        raise AssertionError("image failed quality gate: " + ", ".join(reasons))
