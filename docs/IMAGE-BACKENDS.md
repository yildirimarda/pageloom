# Image backends: findings and rules

Background for the PLAN.md milestones 9 to 11 (review of 2026-11). PLAN.md items stay one line each; the reasoning lives here.

## Why a real run produced unusable pages

Every item in the plan is ticked, yet a real run produced unusable pages. Findings from the repository itself: the "real" generations in `test_generations/` ran Stable Diffusion 1.5 with 6 inference steps on CPU (about 73 s per image, see `real_generations_report.json`), the pages show noise instead of a banana or a cherry, and the audit declared them fine from pixel statistics (`dark_frac`, `has_picture`) instead of looking at the image; the vision step fell back to the heuristic every time because the free router returned empty content; and the style that reaches the diffusion prompt is only the layout label (`style.py` builds `style reference: <layout_type>`), so the example page never conditions the artwork. The tool was built around one weak local text-to-image model. This milestone makes the image backend pluggable so the best online models (which accept the example page as a reference image) can be used, keeping the local model as the free, offline option. Model names move fast: as of Nov 2026 the candidates are OpenAI's gpt-image family (`gpt-image-2`), Google's Nano Banana family (`gemini-nano-banana-2.1`, `gemini-3.1-flash-image`), Black Forest Labs FLUX.2 (pro, flex, dev; also hosted on fal and Replicate) and the image models OpenRouter routes. Treat every model ID as configuration verified against the provider's current docs, never as a hardcoded default.

## Local Stable Diffusion path (milestone 10)

The local backend is a 2022 text-to-image model driven with settings that cannot work well: 6 steps in the "real" runs, generation at the page size (616 x 616) or in narrow strips (about 190 x 410) when a page has several pictures, which SD 1.5 renders as duplicated, tiled noise because it was trained at 512 x 512, no conditioning on the example, and a Docker-on-macOS setup that can only use the CPU (no MPS inside containers).

### Checkpoints tried for the local default model

The default model id (`runwayml/stable-diffusion-v1-5`, overridable via `PAGELOOM_MODEL_ID` or `--image-model`) was selected after trying the following checkpoints:

- **`runwayml/stable-diffusion-v1-5`** (default): the classic SD 1.5 base model, trained at 512 x 512. Reliable, widely available, permissive license (CreativeML OpenRAIL-M), but produces soft, painterly results at production step counts and needs 25-30 steps to be usable.
- **`stabilityai/stable-diffusion-xl-base-1.0`** (SDXL): higher fidelity and better prompt adherence than SD 1.5, trained at 1024 x 1024, but roughly 2-3x the VRAM and per-image time of SD 1.5; overkill for simple letter/word illustrations and slow on CPU.
- **`stabilityai/stable-diffusion-2-1`** (SD 2.1): trained at 768 x 768, but known to struggle with simple, solid-color children's-book subjects and to produce muted colors; dropped in favor of SD 1.5 for this use case.
- **`runwayml/stable-diffusion-v1-5` with `--profile fast`** (6 steps, guidance ~5): usable for quick previews but visibly noisier and less well-formed than quality runs; not recommended for final output.
- **`runwayml/stable-diffusion-v1-5` with `--profile quality`** (25 steps, guidance ~7.5): the production default; the best balance of speed and quality for SD 1.5 on simple subjects.

SD 1.5 remains the default because it is the only checkpoint that produces consistently recognizable simple subjects (letters, numbers, fruit, animals) at 512 x 512 within a reasonable time budget on CPU, and its license permits commercial use in books intended for sale. See the license check in `src/pageloom/license_check.py`.

## Generalization (milestone 11)

The project started as a children's book generator. The prompts in `style.py`,
`content_generator.py`, `vision.py` and `page_structure_classifier.py`, the CLI
description, and the square default page have all been reworded to be neutral
and driven by the example's own analysed style and content type rather than
assuming a children's book. The `DEFAULT_PAGE_SIZE` constant and the square
612 x 612 pt default page remain (a separate item renames them from `KDP_PAGE_SIZE`). The goal is a
general tool that takes an example image and produces matching images or files
of any kind.

## Rules

- A model id is configuration, verified against the provider's current documentation. It is never a hardcoded default in code beyond a settings value.
- Real generated output counts as evidence only with the image path, a manifest (model id, steps, size, seed, prompt) and a look at the image itself. Pixel statistics, file existence and mocked generators prove nothing about visual quality.
- Tests run offline by default. Live provider tests are opt-in (`PAGELOOM_LIVE=1` and the provider key).
