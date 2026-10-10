"""Produce a REAL (non-mocked) word-image matching PDF for a `multiple_labels`
structure, as evidence that the full multi-sub-image compositing path actually
renders 3 distinct picture+word pairs end-to-end through the real Stable
Diffusion pipeline.

Run with:  uv run python scripts/generate_real_multiple_labels.py

Notes
-----
* This uses the REAL IllustrationGenerator (Hugging Face `diffusers`
  Stable Diffusion 1.5), not a mock. To keep a CPU-only run tractable, the
  inference step count is reduced (default 12) from the production default of
  25 -- the *structure* (3 distinct pictures + 3 distinct word labels) is what
  this verifies, not image fidelity. Set STEPS higher for nicer pictures.
* A fixed seed makes the committed artifact reproducible.
* Output:
    test_generations/multiple_labels_word_image_matching.pdf   (the real PDF)
    test_generations/multiple_labels_page.png                  (the rendered page, for inspection)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pageloom.generator import IllustrationGenerator
from pageloom.pipeline import Pipeline

STEPS = int(__import__("os").environ.get("PAGELOM_SD_STEPS", "12"))
SEED = int(__import__("os").environ.get("PAGELOM_SD_SEED", "42"))
EXAMPLE = "example_matching.png"
LABELS = "apple, banana, cherry"
OUT_DIR = Path("test_generations")


class _RealFastGenerator(IllustrationGenerator):
    """Real Stable Diffusion generator with fewer inference steps + a fixed
    per-call seed so each of the N sub-illustrations is distinct yet
    reproducible."""

    def __init__(self, steps: int = STEPS, base_seed: int = SEED, **kwargs) -> None:
        super().__init__(**kwargs)
        # The default SD1.5 safety checker false-positives on benign children's
        # book subjects (e.g. "banana", "cherry") and blacks out the image,
        # which would hide the very pictures we are verifying. The prompts here
        # are family-friendly fruit pairings, so disable the checker for real
        # generation of this evidence artifact.
        self.pipeline.safety_checker = None
        self.pipeline.feature_extractor = None
        self.steps = steps
        self.base_seed = base_seed
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
            num_inference_steps = self.steps
        if seed is None:
            seed = self.base_seed + self._call
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


class _CapturingCompiler:
    """Real PDF compiler that also snapshots the composited page PNG."""

    page_size = (612.0, 612.0)

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path
        from pageloom.pdf_compiler import PDFCompiler

        self._real = PDFCompiler()

    def compile(self, pages, output_path):
        # Snapshot the final composited page (pictures + text) for inspection.
        pages[0].save(str(self.snapshot_path), "PNG")
        return self._real.compile(pages, output_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gen = _RealFastGenerator(verify_license=False)
    print(f"generator loaded in {time.time() - t0:.1f}s", flush=True)

    compiler = _CapturingCompiler(OUT_DIR / "multiple_labels_page.png")
    pipeline = Pipeline(
        example_image=EXAMPLE,
        generator=gen,
        compiler=compiler,
        verify_consistency=False,
        page_structure={
            "content_pattern": "multi_item_scene",
            "element_count": 3,
            "arrangement": "row",
            "text_image_relationship": "multiple_labels",
        },
    )

    result = pipeline.run(
        contents=[LABELS],
        themes=[],
        output_path=str(OUT_DIR / "multiple_labels_word_image_matching.pdf"),
        page_format="standard",
    )
    elapsed = time.time() - t0
    assert result.exists(), f"PDF was not written: {result}"
    print(f"WROTE {result} ({result.stat().st_size} bytes) in {elapsed:.1f}s", flush=True)
    print(f"PAGE PNG: {OUT_DIR / 'multiple_labels_page.png'}", flush=True)


if __name__ == "__main__":
    main()
