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
docker run --rm -v $(pwd)/example.png:/app/example.png -v $(pwd)/outputs:/app/outputs pageloom
```

Notes:
- The same `Dockerfile` runs identically on CPU or CUDA; set `CUDA_VISIBLE_DEVICES` as needed.
- The container uses `python:3.12-slim` and installs `gcc`, `libjpeg-dev`, and `zlib1g-dev` for Pillow.
- Dependencies are installed via `uv sync --no-dev` from `pyproject.toml` and `uv.lock`.

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

- `HF_HOME` — Hugging Face cache/home directory (useful in Docker to persist model weights)
- `CUDA_VISIBLE_DEVICES` — restrict CUDA devices in Docker
- `MODEL_ID` — can be referenced in custom scripts to override the default model

## CLI Usage Examples

PageLoom is primarily used as a library via the `Pipeline` class in Python scripts or notebooks.

### Basic Python usage

```python
from pageloom.pipeline import Pipeline
from pathlib import Path

pipeline = Pipeline(example_image="example.png")
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
