"""Page-structure classifier that determines content pattern from an example page."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


class PageStructureClassifier:
    """Classify the structural content pattern of a children's book page."""

    def __init__(
        self,
        source: str | Path | Image.Image | None = None,
        description: str = "",
    ) -> None:
        self.image: Image.Image | None = None
        if isinstance(source, (str, Path)):
            self.image = Image.open(str(source))
        elif isinstance(source, Image.Image):
            self.image = source.copy()
        self.description = description

    def classify(self, image: Image.Image | None = None) -> dict:
        """Analyze the given image (or the source image) and return structural attributes."""
        target = image if image is not None else self.image
        if target is None:
            raise ValueError("No image provided for classification.")

        width, height = target.size
        aspect = width / max(height, 1)

        # Always use description for pattern inference if available; image
        # heuristics supplement when the description is absent or ambiguous.
        desc = (self.description or "").lower()
        if "object" in desc or "name" in desc or "naming" in desc:
            content_pattern = "object_name_pair"
        elif (
            "scene" in desc
            or "multi" in desc
            or "grid" in desc
            or "puzzle" in desc
            or "matching" in desc
        ):
            content_pattern = "multi_item_scene"
        elif "letter" in desc or "alphabet" in desc or "word" in desc:
            content_pattern = "single_letter_word_illustration"
        else:
            # Fallback to image heuristics when no description pattern matches.
            if aspect > 1.5 or width > 700:
                content_pattern = "multi_item_scene"
            else:
                content_pattern = "single_letter_word_illustration"

        # Structural attributes derived from heuristics
        if content_pattern == "object_name_pair":
            element_count = 1
            arrangement = "centered"
            text_image_relationship = "label_per_image"
        elif content_pattern == "multi_item_scene":
            element_count = max(2, int(width / 200))
            arrangement = "grid"
            text_image_relationship = "label_per_image"
        else:
            element_count = 1
            arrangement = "centered"
            text_image_relationship = "label_per_image"

        return {
            "content_pattern": content_pattern,
            "element_count": element_count,
            "arrangement": arrangement,
            "text_image_relationship": text_image_relationship,
            "image_size": (width, height),
        }


def detect_structure(
    source: str | Path | Image.Image,
    description: str = "",
    vision_analyzer: object | None = None,
) -> dict:
    """Detect page structure, preferring a vision-capable LLM analysis when
    available and falling back to the description/heuristic classifier.

    `vision_analyzer` is any object exposing an `analyze(source, description)`
    method that returns a structure dict (see `pageloom.vision.PageVisionAnalyzer`)
    and raises `pageloom.vision.VisionAnalysisError` on failure. Vision failures
    are caught here and reported to stderr as a warning -- they never abort the
    run, since structure detection always has the heuristic classifier as a
    working (if less accurate) fallback.
    """
    if vision_analyzer is not None:
        from pageloom.vision import VisionAnalysisError

        try:
            return vision_analyzer.analyze(source, description=description)
        except VisionAnalysisError as exc:
            exc_text = str(exc).lower()
            vision_capability_keywords = [
                "image",
                "vision",
                "multimodal",
                "does not support",
                "doesn't support",
                "not support",
            ]
            is_vision_capability_mismatch = any(kw in exc_text for kw in vision_capability_keywords)
            warning_msg = (
                f"Warning: vision-based page analysis failed, falling back to "
                f"heuristic classification: {exc}"
            )
            if is_vision_capability_mismatch:
                warning_msg += (
                    " The selected model/provider does not appear to support image input. "
                    "Set OPENROUTER_VISION_MODEL to a vision-capable model, or unset it to use the free auto-router (openrouter/free), which selects one automatically -- see README for how to find vision-capable models."
                )
            print(warning_msg, file=sys.stderr)
    return PageStructureClassifier(source=source, description=description).classify()
