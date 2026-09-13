# Plan

## Milestone 1: Project scaffolding and core dependencies
- [x] Add pyproject.toml with diffusers, Pillow, pypdf, pytest, ruff; include MPS/CPU/CUDA backend config notes
- [x] Create src/pageloom/ package with __init__.py and module stubs for pipeline stages
- [x] Write placeholder test in tests/test_placeholder.py that passes with pytest
- [x] Add README.md with one-paragraph PageLoom summary describing the KDP pipeline
- [x] Write a detailed README.md covering installation (Docker and local via plain Python/uv), model/dependency setup, CLI usage examples, expected output structure, and troubleshooting notes

## Milestone 2: Illustration generation and text overlay
- [x] Implement Stable Diffusion illustration generator using Hugging Face diffusers library directly in Python
- [x] Create style-prompt template builder that takes an example page description and produces a consistent prompt
- [x] Add Pillow-based text overlay layer for letter and word rendering on generated illustrations
- [x] Build print-ready PDF compiler that combines pages into a single KDP-sized PDF
- [x] Add a CLI entrypoint (e.g. `pageloom generate --pages N`) so the number of pages to generate is a runtime parameter instead of hardcoded
- [x] Handle `--pages N` logic: generate the first N items when N is within the available content list, and fall back gracefully (or raise a clear error) when N exceeds it

## Milestone 3: Style consistency and pipeline portability
- [x] Implement visual-style extractor that analyzes an example page for color palette, typography, and composition rules
- [x] Add consistency guard that verifies generated pages match the example's illustration style and layout logic
- [x] Ensure Docker container runs identically on CPU or CUDA with the same Python codebase and no behavioral differences
- [x] Verify macOS MPS backend acceleration runs the same pipeline via plain Python or uv with no code changes
- [x] Verify the chosen Stable Diffusion model's license (e.g. CreativeML OpenRAIL-M) permits commercial use before generated images go into a book intended for sale
- [x] Add caching/resume support so already-generated raw illustrations are skipped on rerun unless a `--force` flag is passed (saves time during iteration)
- [x] Support environment-variable overrides for key settings (model id, device, output dir, HF cache dir) so Docker and local runs can differ in config without code changes
- [x] Document and handle first-time model download/caching (e.g. HF_HOME volume) so Docker builds don't re-download weights on every run
- [x] Expand test coverage with unit tests for the style-prompt builder, text overlay, and PDF compiler individually, not just the end-to-end integration test
- [x] Add an optional LLM-based content-list generator to supply arbitrary N page topics/words when the requested page count exceeds the built-in alphabet list (ties directly into the `--pages N` feature once pages > 26)
- [ ] Support multiple LLM providers (OpenAI, Anthropic, OpenRouter) for the content-list generator behind 
      a single provider-agnostic interface, selectable via a config value or CLI flag (e.g. `--llm-provider`)
- [ ] Read provider credentials from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, 
      `OPENROUTER_API_KEY`), with a clear startup error if the selected provider's key is missing
- [ ] Add a per-provider default model setting (e.g. config values for OpenAI/Anthropic/OpenRouter model IDs) 
      so the model used isn't hardcoded and can be swapped without code changes
- [ ] Add a fallback/error path if the selected provider's API call fails (timeout, rate limit, invalid key) — 
      fail with a clear message rather than silently defaulting to the built-in alphabet list
- [ ] Add a test (mocked, no real API calls) verifying each of the three providers is called with the 
      correct request shape and that provider switching works via config/CLI
- [ ] Extend the consistency guard to validate structural conformance (page-structure classifier output) 
      in addition to style, once the classifier exists — the guard currently only checks color/typography/
      layout, not whether the generated page matches the example's content pattern      

## Milestone 4: Generalization beyond alphabet pages
- [x] Refactor pipeline to infer layout and style directly from any example page instead of hardcoding letter/word templates
- [x] Add support for alternate page formats such as a photo-realistic object with its name and themed scene pages
- [x] Create end-to-end integration test that feeds one example page and generates a matching multi-page PDF
- [ ] Implement an explicit page-structure classifier: given an example page (image + description), 
      determine its content pattern — e.g. single letter+word+illustration, object-name pairs, 
      multi-item labeled scene, matching/puzzle grid — not just its color palette and typography
- [ ] Extend the extracted template to capture structural attributes: number of illustrated elements 
      per page, their spatial arrangement, and the text-to-image relationship (one label per image, 
      multiple labels, no text, etc.)
- [ ] Route the LLM content generator through the detected page structure, so generated content matches 
      the *type* of page found (e.g. object names for a naming-grid page, single words for a letter page) 
      instead of always assuming letter+word format
- [ ] Add a test with two structurally different example pages (e.g. an alphabet page vs. an 
      object-naming page) and confirm the pipeline produces new pages matching each structure — 
      not just matching style
- [ ] Reconcile the `--pages N` overflow trigger with generalized page structures: the current logic only 
      calls the LLM content generator when N exceeds the built-in 26-letter alphabet list, which doesn't 
      apply to non-alphabet example pages (object-naming or scene pages have no fixed built-in count) — 
      the LLM generator should trigger whenever N exceeds whatever content items are explicitly available, 
      regardless of detected page type
- [ ] Update README.md to document LLM provider configuration (env vars, `--llm-provider` flag) and the 
      supported page structures / how a new structure type would be added — these capabilities were 
      planned after the README was already written and are currently undocumented      

## Discovered
- [x] Add safe image-return guard and error handling to generator for empty pipeline results or failed model loads
<!-- work found while building; appended by the agent -->
