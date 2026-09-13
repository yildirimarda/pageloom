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

Example:

```bash
# Generate 30 pages with Anthropic when the built-in alphabet list (26) is exceeded
pageloom generate --pages 30 --use-llm --llm-provider anthropic --example example.png
```

### Per-Provider Default Models

Each provider has its own default model setting so the model isn't hardcoded in code:

- OpenAI: `gpt-4o-mini` (set via `PAGELOOM_LLM_MODEL`)
- Anthropic: `claude-3-haiku-20240307` (set via `ANTHROPIC_LLM_MODEL`)
- OpenRouter: `openai/gpt-4o-mini` (set via `OPENROUTER_LLM_MODEL`)

These can be overridden per provider through the environment variables above or by passing `model` explicitly to `LLMContentGenerator`.

### Error Handling

If an LLM API call fails — for example due to a rate limit, timeout, invalid key, or truncated response — `LLMContentGenerator.generate()` raises a `RuntimeError` with a message identifying the provider and the specific failure type (rate limit, timeout, invalid/missing key, or insufficient content). The CLI catches these and prints an error to stderr (`Error: LLM content generation failed: ...`) rather than silently defaulting to the built-in content list.

## Supported Page Structures

`PageStructureClassifier` determines the structural content pattern of an example page. Three structures are currently supported:

- `single_letter_word_illustration` — a single letter/word with an illustration (default for standard alphabet-style pages). Content is routed through a single-item centered layout (`element_count: 1`, `arrangement: centered`, `text_image_relationship: label_per_image`).
- `object_name_pair` — a photo-realistic object with its name label (`element_count: 1`, `arrangement: centered`, `text_image_relationship: label_per_image`).
- `multi_item_scene` — a themed scene with multiple labeled items, such as a grid or puzzle (`element_count` inferred from image width, `arrangement: grid`, `text_image_relationship: label_per_image`).

The detected structure is passed to `LLMContentGenerator.generate()` via the `page_structure` parameter so generated content matches the page type (e.g., object names for `object_name_pair`, scene descriptions for `multi_item_scene`). Additionally, `Pipeline.run()` derives the default `page_format` (`object_name` or `themed_scene`) from the detected structure when `page_format="standard"` is not explicitly overridden.

### Adding a New Structure Type

To add a new page structure:

1. Update `PageStructureClassifier.classify()` in `src/pageloom/page_structure_classifier.py` to detect the new pattern (e.g., from the image description or visual heuristics) and return a new `content_pattern` value along with the appropriate `element_count`, `arrangement`, and `text_image_relationship`.
2. Add a new prompt branch in `LLMContentGenerator.generate()` in `src/pageloom/content_generator.py` for the new `content_pattern`, defining how content words/topics should be generated for that structure.
3. Update this README to document the new structure and its structural attributes.
4. Add a test in `tests/test_content_generator.py` verifying the new pattern produces content with the correct request shape (similar to `test_llm_content_generator_routed_by_object_name_pair`).

### Description-Based Pattern Inference

The pipeline, extractor, classifier, and consistency guard all accept an optional `description` string. When a page is described (e.g., `"object name pair"` or `"alphabet letter word"`), the description guides `PageStructureClassifier` to infer the correct content pattern even when the image dimensions alone would suggest a different structure. Pass `description` to `Pipeline`, `VisualStyleExtractor`, `ConsistencyGuard`, or `PageStructureClassifier` directly to improve structural routing accuracy.

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
result = pipeline.run(
    contents=["Apple", "Banana", "Cherry"],
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
