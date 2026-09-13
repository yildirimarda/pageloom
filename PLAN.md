# Plan

## Milestone 1: Project scaffolding and core dependencies
- [x] Add pyproject.toml with diffusers, Pillow, pypdf, pytest, ruff; include MPS/CPU/CUDA backend config notes
- [x] Create src/pageloom/ package with __init__.py and module stubs for pipeline stages
- [x] Write placeholder test in tests/test_placeholder.py that passes with pytest
- [x] Add README.md with one-paragraph PageLoom summary describing the KDP pipeline
- [ ] Write a detailed README.md covering installation (Docker and local via plain Python/uv), model/dependency setup, CLI usage examples, expected output structure, and troubleshooting notes

## Milestone 2: Illustration generation and text overlay
- [x] Implement Stable Diffusion illustration generator using Hugging Face diffusers library directly in Python
- [x] Create style-prompt template builder that takes an example page description and produces a consistent prompt
- [x] Add Pillow-based text overlay layer for letter and word rendering on generated illustrations
- [x] Build print-ready PDF compiler that combines pages into a single KDP-sized PDF
- [ ] Add a CLI entrypoint (e.g. `pageloom generate --pages N`) so the number of pages to generate is a runtime parameter instead of hardcoded
- [ ] Handle `--pages N` logic: generate the first N items when N is within the available content list, and fall back gracefully (or raise a clear error) when N exceeds it

## Milestone 3: Style consistency and pipeline portability
- [x] Implement visual-style extractor that analyzes an example page for color palette, typography, and composition rules
- [x] Add consistency guard that verifies generated pages match the example's illustration style and layout logic
- [x] Ensure Docker container runs identically on CPU or CUDA with the same Python codebase and no behavioral differences
- [x] Verify macOS MPS backend acceleration runs the same pipeline via plain Python or uv with no code changes
- [ ] Verify the chosen Stable Diffusion model's license (e.g. CreativeML OpenRAIL-M) permits commercial use before generated images go into a book intended for sale
- [ ] Add caching/resume support so already-generated raw illustrations are skipped on rerun unless a `--force` flag is passed (saves time during iteration)
- [ ] Support environment-variable overrides for key settings (model id, device, output dir, HF cache dir) so Docker and local runs can differ in config without code changes
- [ ] Document and handle first-time model download/caching (e.g. HF_HOME volume) so Docker builds don't re-download weights on every run
- [ ] Expand test coverage with unit tests for the style-prompt builder, text overlay, and PDF compiler individually, not just the end-to-end integration test
- [ ] Add an optional LLM-based content-list generator to supply arbitrary N page topics/words when the requested page count exceeds the built-in alphabet list (ties directly into the `--pages N` feature once pages > 26)

## Milestone 4: Generalization beyond alphabet pages
- [x] Refactor pipeline to infer layout and style directly from any example page instead of hardcoding letter/word templates
- [x] Add support for alternate page formats such as a photo-realistic object with its name and themed scene pages
- [x] Create end-to-end integration test that feeds one example page and generates a matching multi-page PDF

## Discovered
- [x] Add safe image-return guard and error handling to generator for empty pipeline results or failed model loads
<!-- work found while building; appended by the agent -->
