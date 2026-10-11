"""Evidence that a `multiple_labels` page (a 3-item word-image matching page)
produces a REAL generated PDF showing three distinct picture+word pairs.

The audit (2026-10) found that the only evidence for the `multiple_labels`
compositing path was a mocked test (MagicMock generator) -- `test_generations/`
was empty and no real PDF existed. This module closes that gap:

* A real, non-mocked Stable Diffusion PDF + rendered page are committed under
  ``test_generations/`` (produced by ``scripts/generate_real_multiple_labels.py``).
* The always-on tests below inspect that committed real artifact and assert it
  really contains three distinct picture regions and three distinct word labels
  laid out as picture+word pairs (labels in a bottom band, pictures above).
* ``test_real_sd_generation_multiple_labels_three_pairs`` regenerates a real
  page end-to-end through the real SD pipeline when ``PAGELOM_RUN_REAL_SD=1``
  is set, so the proof can be re-exercised by anyone with the model available.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "example_matching.png"
OUT_DIR = ROOT / "test_generations"
ARTIFACT_PDF = OUT_DIR / "multiple_labels_word_image_matching.pdf"
ARTIFACT_PNG = OUT_DIR / "multiple_labels_page.png"
LABELS = ["apple", "banana", "cherry"]


# ---------------------------------------------------------------------------
# Helpers shared by the always-on (artifact-inspection) tests.
# ---------------------------------------------------------------------------


def _style_and_overlay() -> object:
    from pageloom.overlay import TextOverlay
    from pageloom.style_extractor import VisualStyleExtractor

    style = VisualStyleExtractor(str(EXAMPLE)).extract()
    return TextOverlay(style_profile=style)


def _text_color_mask(
    arr: np.ndarray, text_color: tuple[int, int, int], tol: int = 40
) -> np.ndarray:
    """Boolean mask of pixels close to the label text color."""
    return (
        (np.abs(arr[:, :, 0].astype(int) - text_color[0]) < tol)
        & (np.abs(arr[:, :, 1].astype(int) - text_color[1]) < tol)
        & (np.abs(arr[:, :, 2].astype(int) - text_color[2]) < tol)
    )


def _word_recall(
    page: Image.Image,
    overlay: object,
    word: str,
    position: tuple[int, int],
    tol: int = 40,
) -> float:
    """Return the recall of a word's rendered glyph in the page.

    Re-renders ``word`` with the overlay's own font/color/size at the exact
    centered position the overlay uses, then measures how much of that glyph's
    dark pixels are also dark in the page. A recall near 1.0 means the word is
    actually rendered there (proves distinct *words*, not just 3 dark blobs).
    """
    arr = np.asarray(page.convert("RGB")).astype(int)
    text_color = tuple(int(v) for v in overlay.text_color)  # type: ignore[attr-defined]
    page_dark = _text_color_mask(arr, text_color, tol)

    cx, cy = position
    font = overlay._get_font(overlay.text_size)  # type: ignore[attr-defined]
    ref = Image.new("RGB", page.size, "white")
    d = ImageDraw.Draw(ref)
    bb = font.getbbox(word)
    if bb is None:
        x, y = cx, cy
    else:
        gw, gh = bb[2] - bb[0], bb[3] - bb[1]
        x = cx - gw // 2
        y = cy - gh // 2
    d.text((x, y), word, fill=text_color, font=font)
    ref_dark = _text_color_mask(np.asarray(ref).astype(int), text_color, tol)
    total = int(ref_dark.sum())
    if total == 0:
        return 0.0
    overlap = int(np.logical_and(ref_dark, page_dark).sum())
    return overlap / total


# ---------------------------------------------------------------------------
# Always-on: inspect the committed REAL artifact (no model, no network).
# ---------------------------------------------------------------------------


def test_real_artifact_files_exist():
    assert EXAMPLE.exists(), "example_matching.png missing"
    assert ARTIFACT_PNG.exists(), (
        "real multiple_labels page PNG missing -- run "
        "`uv run python scripts/generate_real_multiple_labels.py` to regenerate"
    )
    assert ARTIFACT_PDF.exists(), (
        "real multiple_labels PDF missing -- run "
        "`uv run python scripts/generate_real_multiple_labels.py` to regenerate"
    )
    from pypdf import PdfReader

    reader = PdfReader(str(ARTIFACT_PDF))
    assert len(reader.pages) == 1
    box = reader.pages[0].mediabox
    assert float(box.width) > 0 and float(box.height) > 0


def test_artifact_shows_three_distinct_picture_regions():
    """The top 70% of the page holds three visually distinct illustrations."""
    page = Image.open(ARTIFACT_PNG).convert("RGB")
    arr = np.asarray(page).astype(int)
    H, W, _ = arr.shape
    pic_h = int(H * 0.70)
    colors = []
    for s in range(3):
        strip = arr[0:pic_h, s * W // 3 : (s + 1) * W // 3].reshape(-1, 3)
        # picture pixels = non-white, non-label-dark
        is_pic = ~((strip > 245).all(axis=1)) & ~((strip < 90).all(axis=1))
        pic = strip[is_pic]
        assert len(pic) > 500, f"picture strip {s} is empty/black, not a real illustration"
        colors.append(tuple(int(v) for v in pic.mean(axis=0)))

    # The three illustrations must be visually distinct (not one picture).
    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    d01, d02, d12 = (
        dist(colors[0], colors[1]),
        dist(colors[0], colors[2]),
        dist(colors[1], colors[2]),
    )
    assert d01 > 30 and d02 > 30 and d12 > 30, (
        f"expected three distinct picture colors, got {colors} "
        f"(pairwise {round(d01)},{round(d02)},{round(d12)})"
    )


def test_artifact_shows_three_distinct_word_labels():
    """Three different words (apple/banana/cherry) are rendered as labels."""
    page = Image.open(ARTIFACT_PNG).convert("RGB")
    overlay = _style_and_overlay()
    positions = overlay._multiple_label_positions(  # type: ignore[attr-defined]
        page.width, page.height, len(LABELS), arrangement="row"
    )
    # All three words must be present at the three label positions (recall > .5),
    # and each position must be a different word -> not one label or duplicated text.
    seen_words = []
    for word, pos in zip(LABELS, positions):
        recall = _word_recall(page, overlay, word, pos)
        assert recall > 0.5, (
            f"word '{word}' not rendered at its label position (recall={recall:.2f})"
        )
        seen_words.append(word)
    assert sorted(seen_words) == sorted(LABELS), f"expected {LABELS}, got {seen_words}"
    # And the three label positions must be separated (not overlapping each other).
    xs = sorted(p[0] for p in positions)
    assert xs[1] - xs[0] > 60 and xs[2] - xs[1] > 60, f"label positions too close: {positions}"


def test_labels_sit_in_bottom_band_not_over_pictures():
    """Labels occupy the bottom band only; pictures occupy the top -- no overlap."""
    page = Image.open(ARTIFACT_PNG).convert("RGB")
    H = page.height
    pic_h = int(H * 0.70)
    overlay = _style_and_overlay()
    positions = overlay._multiple_label_positions(  # type: ignore[attr-defined]
        page.width, H, len(LABELS), arrangement="row"
    )
    # Every label sits below the picture area (no label overlaps a picture).
    assert all(cy >= pic_h for _cx, cy in positions), (
        f"label(s) overlap the picture area: {positions} (pic_h={pic_h})"
    )


# ---------------------------------------------------------------------------
# On-demand: a fully real (non-mocked) end-to-end generation through SD.
# Gated so the default suite stays fast and CI (keyless, no model) is green;
# set PAGELOM_RUN_REAL_SD=1 (model available) to exercise the real pipeline.
# ---------------------------------------------------------------------------

runs_real_sd = pytest.mark.skipif(
    os.environ.get("PAGELOM_RUN_REAL_SD") not in ("1", "true", "TRUE"),
    reason="set PAGELOM_RUN_REAL_SD=1 to run real Stable Diffusion generation",
)


@runs_real_sd
def test_real_sd_generation_multiple_labels_three_pairs(tmp_path):
    """End-to-end real SD generation of a multiple_labels page -> 3 pairs."""
    from pageloom.generator import IllustrationGenerator
    from pageloom.pipeline import Pipeline, round_to_multiple

    class _FastSDGenerator(IllustrationGenerator):
        """Real IllustrationGenerator with fewer steps + safety off for fast CPU runs."""

        def __init__(self, steps: int = 8, base_seed: int = 1234) -> None:
            super().__init__(verify_license=False)
            # safety_checker is handled by the base IllustrationGenerator
            # at from_pretrained time; no clear needed here.
            self._steps = steps
            self._base_seed = base_seed
            self._call = 0

        def generate(
            self,
            prompt,
            negative_prompt="",
            width=512,
            height=512,
            num_inference_steps=None,
            guidance_scale=7.5,
            seed=None,
        ):
            if num_inference_steps is None:
                num_inference_steps = self._steps
            if seed is None:
                seed = self._base_seed + self._call
            self._call += 1
            return super().generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

    captured: dict = {}

    class _CapCompiler:
        page_size = (612.0, 612.0)

        def compile(self, pages, output_path):
            captured["pages"] = [p.copy() for p in pages]
            import os
            import tempfile
            from pathlib import Path

            from pypdf import PdfReader, PdfWriter

            writer = PdfWriter()
            for p in pages:
                img = p.convert("RGB").resize((612, 612))
                fd, p_ = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                img.save(p_, "PDF", resolution=72.0)
                writer.append(PdfReader(p_))
                os.unlink(p_)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer.write(str(output_path))
            return Path(output_path)

    try:
        gen = _FastSDGenerator(steps=8, base_seed=99)
    except Exception as exc:  # noqa: BLE001 - model not cached / no network
        pytest.skip(f"Stable Diffusion model unavailable: {exc}")

    pipeline = Pipeline(
        example_image=str(EXAMPLE),
        generator=gen,
        compiler=_CapCompiler(),
        verify_consistency=False,
        page_structure={
            "content_pattern": "multi_item_scene",
            "element_count": 3,
            "arrangement": "row",
            "text_image_relationship": "multiple_labels",
        },
    )
    # The pipeline must call generate() exactly 3 times (one illustration per
    # element) for the single content item with element_count=3.
    assert gen._call == 0
    pipeline.run(
        contents=["apple, banana, cherry"],
        themes=[],
        output_path=str(tmp_path / "out.pdf"),
        page_format="standard",
    )
    assert gen._call == 3, f"expected 3 sub-illustrations generated, got {gen._call}"

    page = captured["pages"][0]
    assert page.size == (round_to_multiple(612), round_to_multiple(612))

    # 3 distinct picture regions in the top band.
    arr = np.asarray(page.convert("RGB")).astype(int)
    H, W, _ = arr.shape
    pic_h = int(H * 0.70)
    colors = []
    for s in range(3):
        strip = arr[0:pic_h, s * W // 3 : (s + 1) * W // 3].reshape(-1, 3)
        is_pic = ~((strip > 245).all(axis=1)) & ~((strip < 90).all(axis=1))
        colors.append(tuple(int(v) for v in strip[is_pic].mean(axis=0)))

    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    assert dist(colors[0], colors[1]) > 20
    assert dist(colors[0], colors[2]) > 20
    assert dist(colors[1], colors[2]) > 20

    # 3 distinct word labels rendered at separated bottom-band positions.
    overlay = _style_and_overlay()
    positions = overlay._multiple_label_positions(W, H, 3, arrangement="row")
    assert all(cy >= pic_h for _cx, cy in positions)
    for word, pos in zip(LABELS, positions):
        recall = _word_recall(page, overlay, word, pos)
        assert recall > 0.4, f"real generation: word '{word}' not rendered (recall={recall:.2f})"
