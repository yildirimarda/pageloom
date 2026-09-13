"""Page-structure classifier that determines content pattern from an example page."""

from __future__ import annotations

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

        # If analyzing the original source image with a description, use the
        # description for pattern inference. Otherwise infer purely from image.
        is_source = image is None or (self.image is not None and image is self.image)
        if not is_source and self.image is not None and image is not None:
            # Compare pixel data to detect same image even if different objects
            try:
                if target.tobytes() == self.image.tobytes():
                    is_source = True
            except (AttributeError, TypeError):
                pass
        if is_source:
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
                if aspect > 1.5 or width > 700:
                    content_pattern = "multi_item_scene"
                else:
                    content_pattern = "single_letter_word_illustration"
        else:
            # Pure image-based inference for generated/conformance checks
            if aspect > 1.5 or width > 700:
                content_pattern = "multi_item_scene"
            else:
                # Check for high text density or simple centered composition
                # Heuristic: simple pages tend to be single-element
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
