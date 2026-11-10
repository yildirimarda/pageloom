# Image backends: findings and rules

Background for the PLAN.md milestones 9 to 11 (review of 2026-11). PLAN.md items stay one line each; the reasoning lives here.

## Why a real run produced unusable pages

Every item in the plan is ticked, yet a real run produced unusable pages. Findings from the repository itself: the "real" generations in `test_generations/` ran Stable Diffusion 1.5 with 6 inference steps on CPU (about 73 s per image, see `real_generations_report.json`), the pages show noise instead of a banana or a cherry, and the audit declared them fine from pixel statistics (`dark_frac`, `has_picture`) instead of looking at the image; the vision step fell back to the heuristic every time because the free router returned empty content; and the style that reaches the diffusion prompt is only the layout label (`style.py` builds `style reference: <layout_type>`), so the example page never conditions the artwork. The tool was built around one weak local text-to-image model. This milestone makes the image backend pluggable so the best online models (which accept the example page as a reference image) can be used, keeping the local model as the free, offline option. Model names move fast: as of Nov 2026 the candidates are OpenAI's gpt-image family (`gpt-image-2`), Google's Nano Banana family (`gemini-nano-banana-2.1`, `gemini-3.1-flash-image`), Black Forest Labs FLUX.2 (pro, flex, dev; also hosted on fal and Replicate) and the image models OpenRouter routes. Treat every model ID as configuration verified against the provider's current docs, never as a hardcoded default.

## Local Stable Diffusion path (milestone 10)

The local backend is a 2022 text-to-image model driven with settings that cannot work well: 6 steps in the "real" runs, generation at the page size (616 x 616) or in narrow strips (about 190 x 410) when a page has several pictures, which SD 1.5 renders as duplicated, tiled noise because it was trained at 512 x 512, no conditioning on the example, and a Docker-on-macOS setup that can only use the CPU (no MPS inside containers).

## Generalization (milestone 11)

The project started as a children's book generator and still says so in code: prompts in `style.py`, `content_generator.py`, `vision.py` and `page_structure_classifier.py`, the CLI description, the `KDP_PAGE_SIZE` constant and the square 612 x 612 pt default page. The goal is a general tool that takes an example image and produces matching images or files of any kind.

## Rules

- A model id is configuration, verified against the provider's current documentation. It is never a hardcoded default in code beyond a settings value.
- Real generated output counts as evidence only with the image path, a manifest (model id, steps, size, seed, prompt) and a look at the image itself. Pixel statistics, file existence and mocked generators prove nothing about visual quality.
- Tests run offline by default. Live provider tests are opt-in (`PAGELOOM_LIVE=1` and the provider key).
