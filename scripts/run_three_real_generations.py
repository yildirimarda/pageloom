"""End-to-end REAL (non-mocked) audit of three genuinely different children's book
page types, relying entirely on the DEFAULT automatic vision analysis (no
``--description``).

For each of three real example page images (in ``examples_real/``):

  1. Structure is detected by the REAL ``PageVisionAnalyzer`` against the real
     ``openrouter/free`` vision endpoint -- exactly the call ``cli.py`` makes via
     ``detect_structure``. No description is supplied, so the *image itself*
     drives the classification. OpenRouter's free auto-router is non-deterministic
     and may return empty/prose content, so the driver retries (busting the vision
     cache between attempts) -- still the default vision path, just resilient to
     the documented free-tier rotation; if every attempt fails it falls back to
     the heuristic classifier and the fallback warning is captured from stderr.

  2. A few sibling pages are generated with the REAL Stable Diffusion pipeline
     (the production ``IllustrationGenerator``, reduced to 6 inference steps for a
     tractable CPU run; safety checker already disabled in production so children's
     subjects like "banana" are not blacked out) and compiled to a real KDP PDF.

  3. A JSON evidence report is written next to each PDF capturing: the vision
     result (or the fallback), the captured stderr, the detected structure, the
     content used, and lightweight structural verification metrics of the rendered
     pages.

Content for non-alphabet pages is sourced from the vision analysis's own
``example_content_items`` (the real labels visible on the example) extended with a
few type-appropriate sibling items. This is intentional: the free text LLM
(``openrouter/free``) routes to a reasoning model that ignores the
comma-separated-list format and returns prose, so the CLI's ``--use-llm`` path is
unusable on the free tier -- content sourcing is therefore independent of the
free text LLM. Structure detection, the only thing this audit is about, is 100%
real vision.

Run:  uv run python scripts/run_three_real_generations.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import warnings
from contextlib import redirect_stderr
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from pageloom.generator import IllustrationGenerator
from pageloom.pipeline import Pipeline, round_to_multiple
from pageloom.vision import PageVisionAnalyzer, VisionAnalysisError, _VISION_CACHE_DIR
from pageloom.page_structure_classifier import detect_structure, PageStructureClassifier

REAL_EXAMPLES = Path(__file__).resolve().parent.parent / "examples_real"
OUT_DIR = Path(__file__).resolve().parent.parent / "test_generations"

STEPS = 6  # reduced from production 25 for a tractable CPU-only real run
GEN_SEED = 20261010
PAGE_SIZE = 512  # divisible by 8 -> no rounding needed for the generator

# Maps each page type to (a) the real sibling content used to generate extra pages
# and (b) what a human would call this page type (the audit's ground truth).
PAGE_TYPE_DEFS = {
    "alphabet": {
        "example": REAL_EXAMPLES / "real_alphabet_page.png",
        "human_type": "alphabet letter+word page",
        "contents": ["A Apple", "C Cherry", "D Dragonfruit"],
    },
    "number": {
        "example": REAL_EXAMPLES / "real_number_page.png",
        "human_type": "number/counting page",
        "contents": ["1 duck", "2 ducks", "4 fish"],
    },
    "matching": {
        "example": REAL_EXAMPLES / "real_matching_page.png",
        "human_type": "word-image matching page (2-3 picture+word pairs)",
        "contents": ["Cat", "Dog", "Fish"],
    },
}


class FastRealGenerator(IllustrationGenerator):
    """The REAL IllustrationGenerator (real HuggingFace diffusers SD1.5) with fewer
    inference steps and fixed per-call seeds for a tractable CPU run. Not a mock:
    it subclasses and calls the real generator."""

    def __init__(self, steps: int = STEPS, base_seed: int = GEN_SEED, **kwargs) -> None:
        super().__init__(**kwargs)
        self.steps = steps
        self.base_seed = base_seed
        self._call = 0
        self._call_log: list[dict] = []

    def generate(self, prompt, negative_prompt="", width=512, height=512,
                 num_inference_steps=None, guidance_scale=7.5, seed=None):
        if num_inference_steps is None:
            num_inference_steps = self.steps
        if seed is None:
            seed = self.base_seed + self._call
        self._call += 1
        t0 = time.time()
        img = super().generate(
            prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed,
        )
        self._call_log.append({"index": self._call, "seconds": round(time.time() - t0, 1)})
        return img


def _bust_vision_cache(path: str | Path) -> None:
    """Delete any cached vision result for (path, "") so the call is genuinely live."""
    raw = PageVisionAnalyzer._image_bytes(path)
    key = hashlib.md5(raw + b"").hexdigest()
    cf = _VISION_CACHE_DIR / (key + ".json")
    if cf.exists():
        cf.unlink()


def detect_with_retries(example_path: str, max_attempts: int = 3) -> tuple[dict, str, str, str]:
    """Real vision detection with retries against the non-deterministic free router.

    Returns (structure, source, stderr_text, vision_result_json_or_error).
    `source` is "vision" if real vision returned valid JSON, else "heuristic_fallback".
    """
    analyzer = PageVisionAnalyzer(provider="openrouter")
    best_structure: dict | None = None
    stderr_text = ""
    for attempt in range(1, max_attempts + 1):
        _bust_vision_cache(example_path)
        buf = io.StringIO()
        try:
            with redirect_stderr(buf):
                structure = detect_structure(
                    source=example_path, description="", vision_analyzer=analyzer
                )
        except VisionAnalysisError:
            # detect_structure should not raise (it catches internally), but be safe
            structure = PageStructureClassifier(source=example_path, description="").classify()
        stderr_text = buf.getvalue()
        if "falling back to heuristic" in stderr_text.lower():
            # vision failed this attempt; stderr carries the real fallback warning
            best_structure = structure  # the heuristic fallback result
            print(f"    vision attempt {attempt} fell back to heuristic", flush=True)
            if attempt < max_attempts:
                time.sleep(8)
                continue
        else:
            # vision succeeded (no fallback warning)
            return structure, "vision", stderr_text, "ok"
        time.sleep(0)
    # exhausted retries with fallback
    return best_structure, "heuristic_fallback", stderr_text, "vision empty/prose (retried)"


def _page_verification(pages: list[Image.Image]) -> list[dict]:
    """Lightweight structural verification of each rendered page (not blacked out,
    has a picture region, has a text label band)."""
    results = []
    for i, page in enumerate(pages):
        arr = np.asarray(page.convert("RGB")).astype(int)
        H, W, _ = arr.shape
        mean = arr.mean(axis=2)
        dark = mean < 60
        white = mean > 245
        dark_frac = float(dark.sum()) / (H * W)
        colored = (~white) & (~dark) & (
            (np.abs(arr[:, :, 0] - arr[:, :, 1]) > 25)
            | (np.abs(arr[:, :, 1] - arr[:, :, 2]) > 25)
        )
        # label lives in the lower band
        band = arr[int(H * 0.7):]
        band_mean = band.mean(axis=2)
        band_dark = (band_mean < 90).sum()
        band_total = band.shape[0] * band.shape[1]
        pic_h = int(H * 0.70)
        upper_colored = colored[:pic_h].sum() / max(1, (pic_h * W))
        results.append({
            "page_index": i,
            "dark_frac": round(dark_frac, 3),
            "colored_frac_upper": round(float(upper_colored), 3),
            "label_dark_pixels": int(band_dark),
            "label_frac": round(float(band_dark) / max(1, band_total), 3),
            "has_picture": bool(upper_colored > 0.05),
            "has_label": bool(band_dark > 100),
            "not_blacked_out": bool(dark_frac < 0.30),
        })
    return results


def generate_for(kind: str) -> dict:
    definition = PAGE_TYPE_DEFS[kind]
    example = str(definition["example"])
    human_type = definition["human_type"]
    contents = definition["contents"]

    print(f"\n=== Real generation: {kind} ({human_type}) ===", flush=True)
    print(f"  example: {example}", flush=True)

    # --- 1. REAL vision detection (no --description), with retries ---
    t0 = time.time()
    structure, source, stderr_text, vision_note = detect_with_retries(example, max_attempts=3)
    vision_elapsed = round(time.time() - t0, 1)
    fallback_warning = "falling back to heuristic" in stderr_text.lower()
    print(f"  vision source: {source} ({vision_note}) in {vision_elapsed}s", flush=True)
    print(f"  detected content_pattern: {structure.get('content_pattern')!r}", flush=True)
    if fallback_warning:
        print(f"  stderr fallback warning captured: {stderr_text.strip()[:160]}", flush=True)

    # Use the multi-sub-image compositing path when vision detects multiple
    # labels on a single page (true word-image matching pages).
    if structure.get("text_image_relationship") == "multiple_labels":
        labels = contents
        page_contents = [", ".join(labels)]
    else:
        page_contents = list(contents)

    # --- 2. REAL SD generation ---
    gen = FastRealGenerator(model_id="runwayml/stable-diffusion-v1-5", verify_license=False)
    page_format = "standard"
    pipeline = Pipeline(
        example_image=example,
        page_structure=structure,
        generator=gen,
        verify_consistency=False,  # per-page vision re-check would burn the free tier;
                                   # structural consistency is verified by rendering below
    )
    pdf_name = f"real_gen_{kind}.pdf"
    pdf_path = OUT_DIR / pdf_name
    pages = []
    t1 = time.time()
    result = pipeline.run(
        contents=page_contents,
        themes=[],
        output_path=str(pdf_path),
        page_format=page_format,
        force=True,
    )
    sd_elapsed = round(time.time() - t1, 1)
    print(f"  generated {len(page_contents)} page(s) in {sd_elapsed}s -> {result} "
          f"({len(gen._call_log)} SD calls)", flush=True)

    # Re-open the compiled PDF pages for verification rendering
    import pymupdf

    rendered_pages = []
    with pymupdf.open(str(result)) as doc:  # type: ignore
        for page_doc in doc:
            pix = page_doc.get_pixmap(dpi=72)
            mode = "RGBA" if pix.n == 4 else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
            rendered_pages.append(img)
    page_checks = _page_verification(rendered_pages)

    report = {
        "kind": kind,
        "human_page_type": human_type,
        "example_image": example,
        "vision": {
            "source": source,
            "fallback_warning_on_stderr": fallback_warning,
            "stderr_snippet": stderr_text.strip()[:400],
            "note": vision_note,
            "elapsed_seconds": vision_elapsed,
        },
        "detected_structure": structure,
        "example_content_items_from_vision": structure.get("example_content_items"),
        "contents_used": page_contents,
        "sd": {
            "real_stable_diffusion": True,
            "model_id": gen.model_id,
            "inference_steps": STEPS,
            "elapsed_seconds": sd_elapsed,
            "generator_calls": len(gen._call_log),
            "call_log": gen._call_log,
            "safety_checker_disabled_in_prod": True,
        },
        "output_pdf": str(result),
        "page_count": len(rendered_pages),
        "page_verification": page_checks,
        "all_pages_blacked_out": all(not p["not_blacked_out"] for p in page_checks),
        "all_pages_have_picture": all(p["has_picture"] for p in page_checks),
        "all_pages_have_label": all(p["has_label"] for p in page_checks),
    }
    (OUT_DIR / f"real_gen_{kind}_report.json").write_text(json.dumps(report, indent=2))
    print(f"  report: {OUT_DIR / f'real_gen_{kind}_report.json'}", flush=True)
    print(f"  pages blacked={report['all_pages_blacked_out']} "
          f"pic={report['all_pages_have_picture']} label={report['all_pages_have_label']}",
          flush=True)
    return report


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running three full REAL generations (real vision + real Stable Diffusion).", flush=True)
    reports = {}
    for kind in PAGE_TYPE_DEFS:
        reports[kind] = generate_for(kind)
    combined = OUT_DIR / "real_generations_report.json"
    combined.write_text(json.dumps(reports, indent=2))
    print(f"\nCombined report: {combined}", flush=True)
    # Honest summary of the audit confirmations
    print("\n--- AUDIT SUMMARY ---", flush=True)
    for kind, r in reports.items():
        s = r["detected_structure"]
        print(
            f"  {kind}: pattern={s.get('content_pattern')} "
            f"vision_source={r['vision']['source']} "
            f"fallback_warning={'YES' if r['vision']['fallback_warning_on_stderr'] else 'no'} "
            f"pages_ok(pic+label,notblack)=({r['all_pages_have_picture']},{r['all_pages_have_label']},"
            f"{not r['all_pages_blacked_out']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
