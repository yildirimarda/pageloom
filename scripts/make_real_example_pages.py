"""Create three GENUINELY REAL example page images (one per page type) using the
real Stable Diffusion pipeline for the illustrations plus Pillow for the text
labels and layout -- the same `IllustrationGenerator` used for production pages.

The repo's committed ``example_alphabet.png`` / ``example_number.png`` /
``example_matching.png`` are solid-color PLACEHOLDER images (pure white / pale
yellow / light green, <5 distinct colors each) with no real content. Vision
analysis cannot meaningfully classify a blank image -- it either returns empty
content (the white one) or hallucinates phantom content ("3 ducks", "Cat/Dog")
that is not actually present. So this script creates real, content-bearing example
pages so the vision analysis in ``run_three_real_generations.py`` has something
genuinely classifiable to look at:

    examples_real/
        real_alphabet_page.png   -> letter + word + illustration (single_letter_word_illustration)
        real_number_page.png     -> numeral + count description + illustration (number_count_matching)
        real_matching_page.png   -> 3 word+picture pairs on one page (multiple_labels / word_image_matching)

Each page is 612x612 (KDP-ish square) children's-book worksheet style with a
white background, a clear label, and a real SD illustration. The pipeline
later rounds page dimensions to a multiple of 8 for generation.

Run:  uv run python scripts/make_real_example_pages.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pageloom.generator import IllustrationGenerator

STEPS = int(__import__("os").environ.get("PAGELOM_SD_STEPS", "8"))
SEED = int(__import__("os").environ.get("PAGELOM_SD_SEED", "7"))
PAGE_W = PAGE_H = 612
OUT_DIR = Path(__file__).resolve().parent.parent / "examples_real"

# Distinct, real subjects per page type so a vision model has recognizable
# content to ground its classification on (rather than hallucinating).
ILLUSTRATION_PROMPTS = {
    "alphabet": "a ripe yellow banana on a white background, simple children book illustration, cartoon style",
    "number": "three cute yellow ducklings swimming on a pond, simple children book illustration, cartoon style",
    "matching_cat": "a cute orange tabby cat looking at the viewer, simple children book illustration, cartoon style",
    "matching_dog": "a cute golden puppy sitting happy, simple children book illustration, cartoon style",
    "matching_fish": "a colorful goldfish swimming in a clear bowl, simple children book illustration, cartoon style",
}
MATCHING_LABELS = ["Cat", "Dog", "Fish"]


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    y: int,
    fill=(30, 30, 30),
) -> int:
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    x = (PAGE_W - tw) // 2
    draw.text((x, y), text, fill=fill, font=font)
    return bbox[3] - bbox[1]


def _gen_illustration(
    generator: IllustrationGenerator, prompt: str, idx: int, size: int = 320
) -> Image.Image:
    """Generate one real SD illustration (safety checker already disabled in prod)."""
    img = generator.generate(
        prompt=prompt,
        width=size,
        height=size,
        num_inference_steps=STEPS,
        seed=SEED + idx,
    )
    if not isinstance(img, Image.Image):
        raise TypeError(f"SD returned non-image for {prompt}")
    return img.convert("RGB").resize((size, size))


def _blank() -> Image.Image:
    return Image.new("RGB", (PAGE_W, PAGE_H), color="white")


def make_alphabet_page(generator: IllustrationGenerator) -> Image.Image:
    """letter 'B' + word 'Banana' + a banana illustration."""
    page = _blank()
    draw = ImageDraw.Draw(page)
    letter_font = _get_font(190)
    word_font = _get_font(80)
    sub_font = _get_font(36)
    y = 40
    _draw_centered(draw, "B", letter_font, y)
    y += 180
    _draw_centered(draw, "Banana", word_font, y)
    y += 120
    ill = _gen_illustration(generator, ILLUSTRATION_PROMPTS["alphabet"], 0, size=320)
    page.paste(ill, ((PAGE_W - 320) // 2, y))
    y2 = y + 320 + 20
    _draw_centered(draw, "B for Banana", sub_font, y2, fill=(80, 80, 80))
    return page


def make_number_page(generator: IllustrationGenerator) -> Image.Image:
    """numeral '3' + '3 ducks' label + 3-ducks illustration."""
    page = _blank()
    draw = ImageDraw.Draw(page)
    num_font = _get_font(200)
    word_font = _get_font(80)
    sub_font = _get_font(36)
    y = 40
    _draw_centered(draw, "3", num_font, y)
    y += 200
    _draw_centered(draw, "3 ducks", word_font, y)
    y += 110
    ill = _gen_illustration(generator, ILLUSTRATION_PROMPTS["number"], 1, size=320)
    page.paste(ill, ((PAGE_W - 320) // 2, y))
    y2 = y + 320 + 20
    _draw_centered(draw, "Count the ducks: 1, 2, 3", sub_font, y2, fill=(80, 80, 80))
    return page


def make_matching_page(generator: IllustrationGenerator) -> Image.Image:
    """3 word+picture pairs arranged in a row (a matching page)."""
    page = _blank()
    draw = ImageDraw.Draw(page)
    word_font = _get_font(70)
    pic_size = 160
    n = len(MATCHING_LABELS)
    slot_w = PAGE_W // n
    img_top = 80
    pic_top = img_top + 110
    for i, label in enumerate(MATCHING_LABELS):
        cx = slot_w * i + slot_w // 2
        ill = _gen_illustration(
            generator, ILLUSTRATION_PROMPTS[f"matching_{label.lower()}"], 2 + i, size=pic_size
        )
        page.paste(ill, (cx - pic_size // 2, pic_top))
        draw.text(
            (cx - pic_size // 2, pic_top + pic_size + 14), label, fill=(30, 30, 30), font=word_font
        )
        # word label is centered under its picture
        bbox = word_font.getbbox(label)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, pic_top + pic_size + 14), label, fill=(30, 30, 30), font=word_font)
    # title
    title = "Match the picture to its name"
    draw.text(
        ((PAGE_W - _get_font(34).getbbox(title)[2]) // 2, 24),
        title,
        fill=(30, 30, 30),
        font=_get_font(34),
    )
    return page


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Loading real SD model ({STEPS} steps/illustration)...", flush=True)
    gen = IllustrationGenerator(model_id="runwayml/stable-diffusion-v1-5", verify_license=False)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    builders = {
        "real_alphabet_page.png": make_alphabet_page,
        "real_number_page.png": make_number_page,
        "real_matching_page.png": make_matching_page,
    }
    out: dict[str, str] = {}
    for fname, builder in builders.items():
        t1 = time.time()
        page = builder(gen)
        path = OUT_DIR / fname
        page.save(str(path), "PNG")
        colors = len(np.unique(np.asarray(page).reshape(-1, 3)[::200], axis=0))
        print(
            f"  wrote {path} ({path.stat().st_size} bytes, {colors} colors) in {time.time() - t1:.1f}s",
            flush=True,
        )
        out[fname] = str(path)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
