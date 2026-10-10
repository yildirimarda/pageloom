# PageLoom

PageLoom is a Python automation pipeline for designing and self-publishing children's books on Amazon KDP. It takes one example page — such as an alphabet illustration with a letter and word — and analyzes its visual style, layout, and content pattern, then generates matching pages for the rest of the set while preserving illustration style, color palette, typography, and composition. Illustrations are generated locally via Stable Diffusion using the Hugging Face diffusers library, text is overlaid with Pillow, and pages are compiled into a single print-ready PDF sized for KDP. The codebase is fully portable: it runs identically inside a Docker container (CPU or CUDA) and on a local macOS machine via plain Python or uv using Apple's MPS backend, with no behavioral differences beyond generation speed.

## Installation

### Docker

A `Dockerfile` is included that builds a container with Python 3.12, `uv`, and all runtime dependencies pre-installed. It supports both CPU and CUDA without code changes.

```bash
# Build the image
docker build -t pageloom .

# Run with a local example image and generated output mounted
mkdir -p outputs
# The container entrypoint runs python -m pageloom.pipeline
# Mount a persistent volume for Hugging Face weights so downloads are cached:
docker run --rm \
  -v $(pwd)/example.png:/app/example.png \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.cache/huggingface:/root/.cache/huggingface \
  pageloom
```

Notes:
- The same `Dockerfile` runs identically on CPU or CUDA; set `CUDA_VISIBLE_DEVICES` as needed.
- The container uses `python:3.12-slim` and installs `gcc`, `libjpeg-dev`, and `zlib1g-dev` for Pillow.
- Dependencies are installed via `uv sync --no-dev` from `pyproject.toml` and `uv.lock`.
- **First-time model download / caching:** The image does not bake in model weights (to keep the image small). On first `docker run`, weights are downloaded into `/root/.cache/huggingface` (set by `HF_HOME`). Mount a host volume (`-v ...:/root/.cache/huggingface`) to persist weights across container restarts and avoid re-downloading on every run.

### Local (plain Python or uv)

Requires Python >= 3.11.

**Using uv (recommended):**

```bash
# Clone or download the repo
uv sync
# Optional: include dev dependencies for tests
uv sync --dev
```

**Using plain Python / pip:**

```bash
pip install -e .
# Or install without editable mode
pip install .
```

## Model and Dependency Setup

PageLoom depends on:

- `diffusers>=0.25` — Stable Diffusion pipeline via Hugging Face
- `torch>=2.2` — PyTorch backend (CPU, CUDA, or MPS)
- `pillow>=10.0` — image processing and text overlay
- `pypdf>=4.0` — PDF compilation
- `transformers>=4.40` — model tokenization support

On first run, the default model (`runwayml/stable-diffusion-v1-5`) is downloaded from Hugging Face. You can configure a different model ID via environment variables or by passing it to `IllustrationGenerator`.

Backend configuration (same code on all platforms):

- **MPS (macOS Apple Silicon):** `torch.backends.mps.is_available()` is checked automatically; set `device="mps"` if needed.
- **CPU:** default fallback; `torch.set_num_threads(...)` can increase parallelism.
- **CUDA:** requires a CUDA-capable `torch` build; `device="cuda"` is selected automatically when available.

Environment variables you can set:

- `HF_HOME` / `PAGELOOM_HF_CACHE_DIR` — Hugging Face cache/home directory (useful in Docker to persist model weights). The Dockerfile sets `HF_HOME=/root/.cache/huggingface` by default; mount this path as a volume.
- `PAGELOOM_MODEL_ID` — override the default Stable Diffusion model (`runwayml/stable-diffusion-v1-5`).
- `PAGELOOM_DEVICE` — override the auto-detected device (`cpu`, `cuda`, `mps`).
- `PAGELOOM_OUTPUT_DIR` — default output directory for generated PDFs.
- `CUDA_VISIBLE_DEVICES` — restrict CUDA devices in Docker

## LLM Provider Configuration

PageLoom supports an optional LLM-based content generator for arbitrary page topics when the requested page count exceeds the built-in content list. It works with three providers: OpenAI (`openai`), Anthropic (`anthropic`), and OpenRouter (`openrouter`).

### Environment Variables

- `PAGELOOM_LLM_PROVIDER` — selected provider (`openai`, `anthropic`, `openrouter`; default: `openai`).
- `PAGELOOM_LLM_API_KEY` — generic API key override (used as fallback when the provider-specific key is not set).
- `OPENAI_API_KEY` — OpenAI provider key.
- `ANTHROPIC_API_KEY` — Anthropic provider key.
- `OPENROUTER_API_KEY` — OpenRouter provider key.
- `PAGELOOM_LLM_MODEL` — default OpenAI model ID (`gpt-4o-mini`).
- `ANTHROPIC_LLM_MODEL` — Anthropic model ID (`claude-3-haiku-20240307`).
- `OPENROUTER_LLM_MODEL` — OpenRouter model ID (`openai/gpt-4o-mini`).
- `PAGELOOM_LLM_ENDPOINT` — custom endpoint URL (mainly for OpenRouter or OpenAI-compatible proxies).

If the selected provider's API key is missing at startup, `LLMContentGenerator` raises a clear `RuntimeError` naming the expected environment variables instead of silently falling back.

### CLI Flags

The CLI wrapper (`pageloom generate --pages N`) supports:

- `--use-llm` — enable LLM-based content generation when `--pages` exceeds the available built-in list.
- `--llm-provider {openai,anthropic,openrouter}` — override the provider selected by `PAGELOOM_LLM_PROVIDER`.
- `--description` — optional free-text description of the example page (e.g. `"object name pair"` or `"alphabet letter word"`). This is passed straight through to structure detection and `Pipeline`. Providing it is treated as an explicit override that **skips automatic vision analysis** (see below) — useful if you already know the page type and want to avoid the network call.
- `--no-vision` — disable automatic vision-based page-structure analysis (see next section) and fall back to the image-size heuristic instead.
- `--vision-provider {openai,anthropic,openrouter}` — provider used for automatic vision analysis (default: `openrouter`, free tier).
- `--no-consistency` — skip the automatic per-page consistency check (see "Consistency Checking" below): no guard runs and no `Consistency check: N/M` summary is printed. Use this to avoid the extra vision API call per page that a first-run check costs (mirrors `--no-vision`, for the same rate-cap concern on the free tier).

Example:

```bash
# Generate 30 pages with Anthropic once the built-in content list is exceeded
pageloom generate --pages 30 --use-llm --llm-provider anthropic --example example.png

# Describe the example page so an object-naming layout is classified correctly
pageloom generate --pages 10 --use-llm --example object_example.png --description "object name pair"
```

Note: the LLM generator triggers whenever `--pages` exceeds the number of content items actually available for the *detected* structure — the built-in 35-item list for `single_letter_word_illustration` pages, or 0 for any other structure (which have no fixed built-in list).

### Per-Provider Default Models

Each provider has its own default model setting so the model isn't hardcoded in code:

- OpenAI: `gpt-4o-mini` (set via `PAGELOOM_LLM_MODEL`)
- Anthropic: `claude-3-haiku-20240307` (set via `ANTHROPIC_LLM_MODEL`)
- OpenRouter: `openai/gpt-4o-mini` (set via `OPENROUTER_LLM_MODEL`)

These can be overridden per provider through the environment variables above or by passing `model` explicitly to `LLMContentGenerator`.

### Error Handling

If an LLM API call fails — for example due to a rate limit, timeout, invalid key, or truncated response — `LLMContentGenerator.generate()` raises a `RuntimeError` with a message identifying the provider and the specific failure type (rate limit, timeout, invalid/missing key, or insufficient content). The CLI catches these and prints an error to stderr (`Error: LLM content generation failed: ...`) rather than silently defaulting to the built-in content list.

## Automatic Visual Page-Structure Analysis (Free by Default)

By default — when you don't pass `--description` or `--no-vision` — PageLoom sends the example page image itself to a vision-capable (multimodal) LLM and asks it to describe the page's structure: what type of page it is, how many illustrated elements it has, how they're arranged, and what the text/image relationship looks like. This means PageLoom can recognize **any** page type — numbers, digits, word-image matching grids, shapes, colors, or anything else — without you writing a description, and without being limited to a fixed list of hardcoded categories.

**Why this needs a separate model.** The `IllustrationGenerator` (Stable Diffusion via `diffusers`) used elsewhere in this pipeline is a text-to-image *generation* model: given a text prompt, it produces a new image. It has no ability to look at an existing image and describe what's in it — that capability doesn't exist in a generation-only model, regardless of which Stable Diffusion checkpoint is configured. Analyzing the example page therefore requires a genuinely multimodal (vision-capable) model, which is what `PageVisionAnalyzer` (`src/pageloom/vision.py`) calls out to via the same provider abstraction used by `LLMContentGenerator`.

**How to find current free vision-capable models on OpenRouter.**

OpenRouter's free model roster changes constantly, so rather than hardcoding any specific model ID, the codebase defaults to the `openrouter/free` auto-router (`OPENROUTER_VISION_MODEL`). OpenRouter's own docs state that this router "smartly filters for models that support features needed for your request such as image understanding," which means it should automatically select a vision-capable model for a vision request without the user needing to look anything up.

If you prefer to pick a specific model (e.g., because you want a particular provider's free tier or need a vision-capable model with a larger context window), you have two options:

(a) Browse https://openrouter.ai/models and filter by input modality **"image"** — this shows vision-capable models across all providers (paid and free). You can also visit https://openrouter.ai/collections/vision-models for a maintained list.

(b) As of this writing (Sept 2026), observed free vision-capable models include `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free`, `inclusionai/ling-3.0-flash-vl:free`, and `thinkingmachines/inkling-small:free`. Because OpenRouter's free-model roster "changes constantly" (their own wording), treat any named model as a snapshot with a freshness caveat — always verify the model is still free and vision-capable before relying on it.

Set `OPENROUTER_VISION_MODEL` to the specific model ID you want, or leave it unset to use the free auto-router (`openrouter/free`), which avoids hardcoding a model that might get retired.

**Free by default.** The default provider for vision analysis is OpenRouter (override with `PAGELOOM_VISION_PROVIDER` or `--vision-provider`), using OpenRouter's `openrouter/free` auto-router as the default model (`OPENROUTER_VISION_MODEL`). This router selects a free, zero-cost model that supports image understanding — OpenRouter's free model roster changes over time, so the auto-router avoids hardcoding a specific model ID that might get retired. You only need an `OPENROUTER_API_KEY` (free to create, no credit card required) for this to work; see [openrouter.ai](https://openrouter.ai) to get one. OpenAI (`gpt-4o-mini`, vision-capable) and Anthropic (`claude-3-haiku-20240307`, vision-capable) are also supported if you'd rather use those.

**Graceful, non-fatal fallback.** If vision analysis is unavailable (no API key configured) or fails (rate limit, timeout, malformed response), PageLoom prints a warning to stderr and falls back automatically to the old image-size heuristic (or your `--description`, if you provide one) rather than aborting the run. Vision analysis is a best-effort upgrade, not a hard requirement.

**How it works internally:** `PageVisionAnalyzer.analyze()` base64-encodes the example image, sends it alongside a prompt asking for a JSON structure description, and parses the response into a dict shaped like:

```json
{
  "content_pattern": "number_count_matching",
  "content_type_description": "a page pairing a printed number with a matching count of illustrated objects",
  "element_count": 3,
  "arrangement": "row",
  "text_image_relationship": "label_per_image",
  "example_content_items": ["3 ducks"]
}
```

The `content_pattern` here is not restricted to the three legacy values below — it can be any short identifier the vision model comes up with. `content_type_description` and `example_content_items` (the actual items visible on the example page) are then used by `LLMContentGenerator.generate()` to produce new content matching that exact page type — e.g., `"4 balloons"`, `"5 stars"` for a number-counting page — via a generic, open-ended prompt rather than one of the three hardcoded branches. `ConsistencyGuard` also accepts a `vision_analyzer` so generated pages can be checked against the example using the same open-ended structure, not just the heuristic categories — and `Pipeline.run()` wires this up automatically (see "Consistency Checking" below).

**Vision environment variables:**

- `PAGELOOM_VISION_PROVIDER` — provider used for automatic vision analysis (`openai`, `anthropic`, `openrouter`; falls back to `PAGELOOM_LLM_PROVIDER` if unset, then defaults to `openrouter`).
- `OPENROUTER_VISION_MODEL` — OpenRouter vision model ID (default: `openrouter/free`, the free auto-router).
- `OPENAI_VISION_MODEL` — OpenAI vision model ID (default: `gpt-4o-mini`).
- `ANTHROPIC_VISION_MODEL` — Anthropic vision model ID (default: `claude-3-haiku-20240307`).
- API keys are the same ones used for content generation (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or the generic `PAGELOOM_LLM_API_KEY`).

## Supported Page Structures

`PageStructureClassifier` determines the structural content pattern of an example page using image-size heuristics and description keywords — this is the fallback path used when vision analysis is disabled or unavailable. Three structures are supported by this heuristic path:

- `single_letter_word_illustration` — a single letter/word with an illustration (default for standard alphabet-style pages). Content is routed through a single-item centered layout (`element_count: 1`, `arrangement: centered`, `text_image_relationship: label_per_image`).
- `object_name_pair` — a photo-realistic object with its name label (`element_count: 1`, `arrangement: centered`, `text_image_relationship: label_per_image`).
- `multi_item_scene` — a themed scene with multiple labeled items, such as a grid or puzzle (`element_count` inferred from image width, `arrangement: grid`, `text_image_relationship: label_per_image`).

When vision analysis is used instead (the default), `content_pattern` is open-ended and not limited to these three values — see the section above.

The detected structure is passed to `LLMContentGenerator.generate()` via the `page_structure` parameter so generated content matches the page type (e.g., object names for `object_name_pair`, scene descriptions for `multi_item_scene`, or whatever `content_type_description` says for a vision-derived pattern). Additionally, `Pipeline.run()` derives the default `page_format` (`object_name` or `themed_scene`) from the detected structure when `page_format="standard"` is not explicitly overridden (other/vision-derived patterns keep the generic `standard` format).

### Consistency Checking

Every run automatically verifies the pages it generates against the example. `Pipeline.run()` constructs a `ConsistencyGuard` (`src/pageloom/consistency.py`) from the example's already-extracted style and detected page structure, then calls `verify()` on each finished page (after text overlay, before PDF compilation). The guard compares palette, typography scale, layout type, page dimensions, and structural conformance (content pattern, element count, arrangement, text/image relationship).

The CLI passes the same `vision_analyzer` it used to classify the example into `Pipeline`, so each generated page is re-classified with the same method (vision model or heuristic) as the example — necessary for open-ended vision-derived `content_pattern` values, which the heuristic classifier alone would mismatch. After generation, a one-line summary is printed (`Consistency check: N/M page(s) match the example.`), with one stderr warning per issue on each inconsistent page; a failed check never aborts the run. Results are also available programmatically as `pipeline.consistency_results` — one result dict per generated page with `consistent`, `issues`, and `details` keys.

**Opting out.** With vision enabled, each check re-classifies the generated page — one extra vision API call per page on a first run (the raw-illustration and vision caches only help on reruns), which matters because OpenRouter's free tier is rate-capped. Skip the check entirely with `--no-consistency` on the CLI or `Pipeline(..., verify_consistency=False)` in Python: no guard runs, `consistency_results` stays empty, and no `Consistency check:` report is printed. This mirrors `--no-vision`, which opts out of example analysis for the same cost/reliability reason.

### Adding a New Structure Type

With vision analysis enabled (the default), you generally **don't need to add anything** for a new page type — the vision model describes any page type in plain language via `content_type_description`, and `LLMContentGenerator.generate()` already has a generic prompt path that uses that description directly (see "Automatic Visual Page-Structure Analysis" above).

The steps below are only needed to add a new *hardcoded* pattern to the heuristic fallback path (`PageStructureClassifier`), which matters for `--no-vision` runs or when no vision provider is configured:

1. Update `PageStructureClassifier.classify()` in `src/pageloom/page_structure_classifier.py` to detect the new pattern (e.g., from the image description or visual heuristics) and return a new `content_pattern` value along with the appropriate `element_count`, `arrangement`, and `text_image_relationship`.
2. Add a new prompt branch in `LLMContentGenerator.generate()` in `src/pageloom/content_generator.py` for the new `content_pattern` (only needed if you want a hand-tuned prompt rather than relying on the generic `content_type_description` path).
3. Update this README to document the new structure and its structural attributes.
4. Add a test in `tests/test_content_generator.py` verifying the new pattern produces content with the correct request shape (similar to `test_llm_content_generator_routed_by_object_name_pair`).

### Description-Based Pattern Inference

Automatic vision analysis (above) is the primary way structure is now detected, so you shouldn't need to write a description in normal use. `description` remains supported as a manual override for the heuristic fallback path: the pipeline, extractor, classifier, and consistency guard all accept an optional `description` string, which guides `PageStructureClassifier` to infer the correct content pattern even when the image dimensions alone would suggest a different structure. Pass `description` to `Pipeline`, `VisualStyleExtractor`, `ConsistencyGuard`, or `PageStructureClassifier` directly, or pass `--description` on the CLI (see CLI Flags above) — either way, providing a description skips the vision API call entirely, since it's treated as an explicit override.

## CLI Usage Examples

PageLoom is primarily used as a library via the `Pipeline` class in Python scripts or notebooks.

### Basic Python usage

```python
from pageloom.pipeline import Pipeline
from pathlib import Path

# The pipeline detects the page structure from the example image and routes
# content accordingly. You can provide an optional description for more accurate
# structural inference (e.g., "object name pair" or "alphabet letter word").
pipeline = Pipeline(example_image="example.png", description="alphabet letter word")
# Pass verify_consistency=False to skip the per-page consistency check
# (avoids the extra vision API call per page on a first run).
result = pipeline.run(
    themes=["fruit", "tropical", "orchard"],
    output_path="children_book.pdf",
    page_format="standard",
)
print(f"PDF written to: {result}")
```

### Programmatic CLI-style invocation

```python
from pageloom.generator import IllustrationGenerator
from pageloom.style import StylePromptBuilder
from pageloom.overlay import TextOverlay
from pageloom.pdf_compiler import PDFCompiler
from pageloom.style_extractor import VisualStyleExtractor

# Example: generate a single illustration with style inference
extractor = VisualStyleExtractor("example.png")
style = extractor.extract()

gen = IllustrationGenerator()
prompt_builder = StylePromptBuilder(example_style=style)
prompt = prompt_builder.build(content="Apple", theme="fruit", page_format="standard")
image = gen.generate(prompt=prompt, width=612, height=612)

overlay = TextOverlay(style_profile=style)
final = overlay.apply(image, text="Apple", page_format="standard")

compiler = PDFCompiler()
compiler.compile([final], "apple_page.pdf")
```

### Running the module directly

```bash
python -m pageloom.pipeline
```

The module entrypoint (`python -m pageloom.pipeline`) is available; a CLI wrapper (`pageloom generate --pages N`) is planned in a future milestone.

## Expected Output Structure

After running the pipeline, the output directory should contain:

```
outputs/
  output.pdf          # compiled print-ready KDP PDF (612 x 612 pts, 72 dpi)
  # Intermediate files are cleaned up automatically by PDFCompiler
```

The PDF compiler (`PDFCompiler`) creates temporary `.pdf` files per page and removes them after writing the final output, keeping the workspace clean. The final PDF uses a standard KDP children's book square page size of 612 x 612 points (8.5 in x 8.5 in at 72 dpi).

## Troubleshooting Notes

- **Model download failures:** Ensure `HF_HOME` or the default Hugging Face cache directory is writable. In Docker, mount a volume for persistence: `-v ~/.cache/huggingface:/root/.cache/huggingface`.
- **CUDA out of memory:** The pipeline enables `enable_attention_slicing()` by default to reduce memory usage. If you still hit OOM, lower `width`/`height` or use CPU (`device="cpu"`).
- **Pillow font errors:** `TextOverlay` falls back to `ImageFont.load_default()` if the specified `font_path` is missing or fails. If text appears too small or too large, verify the `style_profile` extracted from your example image includes valid `composition.page_dimensions`.
- **Empty pipeline results:** `IllustrationGenerator.generate()` returns a gray placeholder image (`Image.new("RGB", ...)`) when the model produces no images or the pipeline returns an empty result. Check the prompt and model load status.
- **No behavioral differences across backends:** The same `Pipeline` class and `IllustrationGenerator` settings run identically on CPU, MPS, and CUDA; only generation speed varies.
- **License check:** Before selling generated images, verify the chosen Stable Diffusion model's license (e.g., CreativeML OpenRAIL-M) permits commercial use.
- **First-time model download:** Docker builds do not include model weights by design (to keep image size small). The first `docker run` will download weights into the container; use a persistent volume or `HF_HOME` to avoid re-downloading on every container restart.

## Running Tests

```bash
uv run pytest -q
# Or with plain Python
python -m pytest -q
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```
