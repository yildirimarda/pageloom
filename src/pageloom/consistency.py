"""Consistency guard that verifies generated pages match the example's
illustration style and layout logic."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pageloom.page_structure_classifier import detect_structure
from pageloom.style_extractor import VisualStyleExtractor


class ConsistencyGuard:
    """Verify generated pages against an example page's style and layout."""

    def __init__(
        self,
        example_image: str | Path | Image.Image | None = None,
        example_style: dict | None = None,
        description: str = "",
        color_tolerance_percent: float = 20.0,
        size_tolerance_percent: float = 25.0,
        page_structure: dict | None = None,
        vision_analyzer: object | None = None,
    ) -> None:
        self.example_image: Image.Image | None = None
        self.example_style: dict | None = None
        self.color_tolerance_percent = color_tolerance_percent
        self.size_tolerance_percent = size_tolerance_percent
        self.description = description
        # Only meaningful when there's an example image to (re-)analyze; kept on
        # self so verify() can apply the same detection method (vision or
        # heuristic) to generated images for a like-for-like comparison.
        self.vision_analyzer = vision_analyzer

        if example_image is not None and example_style is not None:
            raise ValueError("Provide either example_image or example_style, not both.")

        if example_image is not None:
            extractor = VisualStyleExtractor(example_image, description=description)
            self.example_style = extractor.extract()
            self.example_image = extractor.image
            self.example_structure = page_structure or detect_structure(
                source=example_image, description=description, vision_analyzer=vision_analyzer
            )
        elif example_style is not None:
            self.example_style = example_style
            layout_hint = str(example_style.get("composition", {}).get("layout_type", ""))
            # No original image available (style-only init) and no vision analysis
            # possible without one -- fall back to the layout hint as a description,
            # same as before, so verify() classifies generated images consistently.
            self.description = description or layout_hint
            if page_structure is not None:
                self.example_structure = page_structure
            else:
                comp = example_style.get("composition", {})
                self.example_structure = {
                    "content_pattern": comp.get("layout_type", "single_letter_word_illustration"),
                    "element_count": 1,
                    "arrangement": "centered",
                    "text_image_relationship": "label_per_image",
                }
        else:
            raise ValueError("Provide either example_image or example_style.")
        if not hasattr(self, "example_image"):
            self.example_image = None

    def verify(self, generated_image: str | Path | Image.Image) -> dict:
        """Compare a generated page to the example and return a result."""
        if isinstance(generated_image, (str, Path)):
            image = Image.open(str(generated_image))
        else:
            image = generated_image.copy()

        generated_style = VisualStyleExtractor(image).extract()
        issues: list[str] = []
        details: dict = {}

        # Palette consistency
        example_palette = self.example_style.get("palette", [])
        generated_palette = generated_style.get("palette", [])
        palette_match = False
        if example_palette and generated_palette:
            # Simple overlap check: count how many generated colors
            # share a quantized bucket with example colors.
            example_colors = [
                tuple(int(x) for x in c.replace("rgb(", "").replace(")", "").split(","))
                for c in example_palette
            ]
            generated_colors = [
                tuple(int(x) for x in c.replace("rgb(", "").replace(")", "").split(","))
                for c in generated_palette
            ]
            overlap = 0
            for ec in example_colors:
                for gc in generated_colors:
                    if max(abs(ec[i] - gc[i]) for i in range(3)) <= 32:
                        overlap += 1
                        break
            palette_match = overlap > 0
        else:
            palette_match = bool(example_palette == generated_palette)
        details["palette_match"] = palette_match
        if not palette_match:
            issues.append("Generated palette does not match example palette.")

        # Typography consistency
        example_typo = self.example_style.get("typography", {})
        generated_typo = generated_style.get("typography", {})
        letter_scale_est = example_typo.get("letter_size_estimate", 0)
        gen_letter_scale = generated_typo.get("letter_size_estimate", 0)
        word_scale_est = example_typo.get("word_size_estimate", 0)
        gen_word_scale = generated_typo.get("word_size_estimate", 0)
        typo_ok = True
        if letter_scale_est > 0:
            letter_diff = abs(gen_letter_scale - letter_scale_est) / letter_scale_est
            typo_ok = typo_ok and (letter_diff <= (self.size_tolerance_percent / 100.0))
        if word_scale_est > 0:
            word_diff = abs(gen_word_scale - word_scale_est) / word_scale_est
            typo_ok = typo_ok and (word_diff <= (self.size_tolerance_percent / 100.0))
        details["typography_match"] = typo_ok
        if not typo_ok:
            issues.append("Typography scale differs from example.")

        # Composition / layout consistency
        example_comp = self.example_style.get("composition", {})
        generated_comp = generated_style.get("composition", {})
        layout_ok = True
        example_layout = example_comp.get("layout_type")
        generated_layout = generated_comp.get("layout_type")
        if example_layout and generated_layout:
            layout_ok = layout_ok and (example_layout == generated_layout)
        details["layout_match"] = layout_ok
        if not layout_ok:
            issues.append(f"Layout type mismatch: {example_layout} vs {generated_layout}.")

        # Page dimension consistency
        example_dims = example_comp.get("page_dimensions", {})
        generated_dims = generated_comp.get("page_dimensions", {})
        dim_ok = True
        if example_dims and generated_dims:
            for key in ("width", "height"):
                ex = example_dims.get(key, 0)
                gen = generated_dims.get(key, 0)
                if ex > 0:
                    dim_diff = abs(gen - ex) / ex
                    dim_ok = dim_ok and (dim_diff <= (self.size_tolerance_percent / 100.0))
        details["dimension_match"] = dim_ok
        if not dim_ok:
            issues.append("Page dimensions differ from example.")

        # Structural conformance (page-structure classifier output). Uses the
        # same detection method (vision analyzer, if configured, else the
        # heuristic classifier) as the example so patterns are comparable --
        # this matters for open-ended vision-derived patterns, which won't
        # match any of the heuristic classifier's fixed 3 categories.
        generated_structure = detect_structure(
            source=image, description=self.description, vision_analyzer=self.vision_analyzer
        )
        example_structure = getattr(self, "example_structure", {})
        structure_ok = True
        structure_issues: list[str] = []
        # Compare content pattern
        example_pattern = example_structure.get("content_pattern")
        generated_pattern = generated_structure.get("content_pattern")
        if example_pattern and generated_pattern and example_pattern != generated_pattern:
            structure_ok = False
            structure_issues.append(
                f"Content pattern mismatch: {example_pattern} vs {generated_pattern}."
            )
        # Compare element count within a reasonable tolerance
        example_elements = example_structure.get("element_count", 1)
        generated_elements = generated_structure.get("element_count", 1)
        if (
            isinstance(example_elements, (int, float))
            and isinstance(generated_elements, (int, float))
            and abs(int(generated_elements) - int(example_elements)) > 1
        ):
            structure_ok = False
            structure_issues.append(
                f"Element count differs: {example_elements} vs {generated_elements}."
            )
        # Compare arrangement
        example_arrangement = example_structure.get("arrangement")
        generated_arrangement = generated_structure.get("arrangement")
        if (
            example_arrangement
            and generated_arrangement
            and example_arrangement != generated_arrangement
        ):
            structure_ok = False
            structure_issues.append(
                f"Spatial arrangement mismatch: {example_arrangement} vs {generated_arrangement}."
            )
        # Compare text-to-image relationship
        example_rel = example_structure.get("text_image_relationship")
        generated_rel = generated_structure.get("text_image_relationship")
        if example_rel and generated_rel and example_rel != generated_rel:
            structure_ok = False
            structure_issues.append(
                f"Text-to-image relationship mismatch: {example_rel} vs {generated_rel}."
            )

        details["structure_match"] = structure_ok
        details["structure_issues"] = structure_issues
        issues.extend(structure_issues)

        consistent = palette_match and typo_ok and layout_ok and dim_ok and structure_ok
        details["consistent"] = consistent
        details["issues"] = issues
        return {
            "consistent": consistent,
            "issues": issues,
            "details": details,
        }
