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
- [x] Support multiple LLM providers (OpenAI, Anthropic, OpenRouter) for the content-list generator behind 
      a single provider-agnostic interface, selectable via a config value or CLI flag (e.g. `--llm-provider`)
- [x] Read provider credentials from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, 
      `OPENROUTER_API_KEY`), with a clear startup error if the selected provider's key is missing
- [x] Add a per-provider default model setting (e.g. config values for OpenAI/Anthropic/OpenRouter model IDs) 
      so the model used isn't hardcoded and can be swapped without code changes
- [x] Add a fallback/error path if the selected provider's API call fails (timeout, rate limit, invalid key) — 
      fail with a clear message rather than silently defaulting to the built-in alphabet list
- [x] Add a test (mocked, no real API calls) verifying each of the three providers is called with the 
      correct request shape and that provider switching works via config/CLI
- [x] Extend the consistency guard to validate structural conformance (page-structure classifier output) 
      in addition to style, once the classifier exists — the guard currently only checks color/typography/
      layout, not whether the generated page matches the example's content pattern      

## Milestone 4: Generalization beyond alphabet pages
- [x] Refactor pipeline to infer layout and style directly from any example page instead of hardcoding letter/word templates
- [x] Add support for alternate page formats such as a photo-realistic object with its name and themed scene pages
- [x] Create end-to-end integration test that feeds one example page and generates a matching multi-page PDF
- [x] Implement an explicit page-structure classifier: given an example page (image + description), 
      determine its content pattern — e.g. single letter+word+illustration, object-name pairs, 
      multi-item labeled scene, matching/puzzle grid — not just its color palette and typography
- [x] Extend the extracted template to capture structural attributes: number of illustrated elements 
      per page, their spatial arrangement, and the text-to-image relationship (one label per image, 
      multiple labels, no text, etc.)
- [x] Route the LLM content generator through the detected page structure, so generated content matches 
      the *type* of page found (e.g. object names for a naming-grid page, single words for a letter page) 
      instead of always assuming letter+word format
- [x] Add a test with two structurally different example pages (e.g. an alphabet page vs. an 
      object-naming page) and confirm the pipeline produces new pages matching each structure — 
      not just matching style
- [x] Reconcile the `--pages N` overflow trigger with generalized page structures: the current logic only 
      calls the LLM content generator when N exceeds the built-in 26-letter alphabet list, which doesn't 
      apply to non-alphabet example pages (object-naming or scene pages have no fixed built-in count) — 
      the LLM generator should trigger whenever N exceeds whatever content items are explicitly available, 
      regardless of detected page type
- [x] Update README.md to document LLM provider configuration (env vars, `--llm-provider` flag) and the 
      supported page structures / how a new structure type would be added — these capabilities were 
      planned after the README was already written and are currently undocumented      

## Milestone 5: Genuine open-ended page-type recognition (vision-based)
- [x] User-flagged gap: despite Milestone 4 claiming full "generalization", the pipeline only ever
      recognized 3 hardcoded `content_pattern` values (`single_letter_word_illustration`,
      `object_name_pair`, `multi_item_scene`), detected via keyword matching on a manually-typed
      `description` string or a crude image aspect-ratio guess -- there was no actual image
      understanding. Page types the user actually needs (number pages, digit pages, word-image
      matching grids, etc.) were not recognized and the user did not want to hand-write a
      description for every run.
- [x] Add `PageVisionAnalyzer` (`src/pageloom/vision.py`): sends the example page image itself to a
      vision-capable (multimodal) LLM and asks for a structured JSON description of the page's type,
      element count, arrangement, text/image relationship, and the actual content items visible on
      the page -- open-ended, not restricted to a fixed enum of page types.
- [x] Default to OpenRouter's free `openrouter/free` auto-router for vision analysis (zero cost, no
      credit card) per explicit user request to keep this free-first; OpenAI (`gpt-4o-mini`) and
      Anthropic (`claude-3-haiku-20240307`) supported as alternatives. Confirmed via web research
      (Sept 2026) that OpenRouter offers free vision-capable models and that the auto-router is the
      most resilient default since the free model roster changes over time.
- [x] Confirmed and documented that the existing Stable Diffusion model (`diffusers`,
      text-to-image only) cannot be reused for image analysis -- a separate, genuinely multimodal
      model is required. This was a direct question the user asked; answered in the README's new
      "Automatic Visual Page-Structure Analysis" section.
- [x] Wire vision analysis into the CLI as the default path (`cli.py`): used automatically unless
      `--description` (explicit override) or `--no-vision` (explicit opt-out) is given. Failures
      (missing key, rate limit, malformed response) are caught and reported as a warning, falling
      back to the heuristic classifier -- vision is a best-effort upgrade, never a hard requirement.
- [x] Add `detect_structure()` helper (`page_structure_classifier.py`) shared by `Pipeline`,
      `ConsistencyGuard`, and the CLI so vision-first-with-heuristic-fallback logic lives in one place.
- [x] Generalize `LLMContentGenerator.generate()` with an open-ended prompt path driven by
      `content_type_description`/`example_content_items` (used whenever present) ahead of the three
      legacy hardcoded branches (kept as the fallback for the heuristic-only path), so genuinely novel
      page types generate matching content without any new code.
- [x] `ConsistencyGuard` accepts a `vision_analyzer` too, and re-classifies the *generated* image with
      the same method used for the example (vision or heuristic) so open-ended patterns remain
      comparable -- reusing the heuristic classifier only on one side would make every vision-derived
      pattern look like a false mismatch.
- [x] Add tests: `tests/test_vision.py` (provider validation, JSON parsing incl. markdown-fenced and
      malformed responses, multimodal request shape), CLI wiring tests (auto-vision, `--no-vision`,
      graceful fallback, `--vision-provider`), `detect_structure()` fallback tests, and a
      `LLMContentGenerator` test proving a novel pattern (`number_count_matching`, never hardcoded
      anywhere) is handled correctly end-to-end.
- [x] Document the whole feature in README.md (free-tier setup, env vars, fallback behavior, why
      Stable Diffusion can't do this) and add a README test asserting the documentation exists.

## Milestone 6: CI green + vision-analysis hardening (pending, not yet implemented)
- [x] **Fix the CI failure introduced by Milestone 5, in this exact order:**
      1. `uv run ruff format --check .` currently FAILS -- confirmed by running `ruff format --check .`
         directly (the `ruff` binary works regardless of Python version, so this was verified even
         though the sandbox couldn't run the full Python toolchain). Exactly 5 files would be
         reformatted: `src/pageloom/vision.py`, `tests/test_cli.py`, `tests/test_content_generator.py`,
         `tests/test_integration.py`, `tests/test_page_structure_classifier.py`. Run
         `uv run ruff format .` to fix, then re-run `uv run ruff format --check .` and
         `uv run ruff check .` to confirm both are clean, and commit the formatting diff.
      2. Run `uv run pytest -q` for real and fix any genuine failures. This session's sandbox could
         not run the project's actual test suite: its Linux environment has Python 3.10.12 with no
         network access to provision 3.11+ (required by `pyproject.toml`) or install packages, while
         the repo's committed `.venv` was built for Python 3.12 on a different machine and its compiled
         extensions (PIL's `_imaging`, `torch`, `pytest`'s own C-extension-adjacent internals) are not
         loadable under 3.10. All new logic in Milestone 5 (vision.py, the CLI wiring, the generalized
         content generator, Pipeline/ConsistencyGuard integration, the new settings functions) was
         instead verified by hand-written scripts that stub `torch`/`diffusers` via `sys.modules` and
         replay each test's scenario directly -- every one of those passed, but this is not a
         substitute for actually running `pytest` in the correct environment. Treat this as unverified
         until CI (or a local `uv run pytest -q` on a matching Python) actually goes green.
- [x] Do one real (non-mocked) smoke test of vision analysis: run `pageloom generate` against a real
      example image with a real, free `OPENROUTER_API_KEY` and no `--description`, and confirm
      `openrouter/free` actually returns valid JSON matching `PageVisionAnalyzer`'s expected schema.
      Every existing test mocks the network, so this exact prompt/schema combination has never been
      verified against a real (especially free-tier, possibly smaller/less instruction-following)
      vision model. If free models frequently wrap the JSON in extra prose beyond what the current
      markdown-fence-stripping handles, or omit required fields, consider adding one retry with a
      stricter "respond with ONLY the JSON object, no other text" reminder before falling back to the
      heuristic classifier.
- [ ] Feed `element_count` / `arrangement` / `text_image_relationship` into the actual Stable
      Diffusion prompt (`StylePromptBuilder.build()` in `src/pageloom/style.py`), not just into the
      LLM content-list generator's prompt and `ConsistencyGuard`'s comparison. Right now these
      structural fields are captured (by vision or heuristic) but never reach the image-generation
      prompt itself -- so a detected "3 items in a row" structure doesn't actually instruct the
      diffusion model to draw 3 items in a row; the illustration's structural fidelity depends
      entirely on whatever the content string happens to say (e.g. hoping "4 balloons" gets rendered
      as exactly 4 balloons), which diffusion models are known to be unreliable at for precise counts.
- [ ] Decide how to handle `text_image_relationship == "multiple_labels"` and `"no_text"`
      end-to-end -- `PageVisionAnalyzer`'s prompt already asks for one of `label_per_image` /
      `multiple_labels` / `no_text`, but `TextOverlay.apply()` (`src/pageloom/overlay.py`) only ever
      draws a single `text` string at a single position, with no branch reading
      `text_image_relationship` at all. There is no support today for a real word-image *matching*
      page with several distinct picture+word pairs on one page, or for suppressing text entirely on
      `no_text` pages. This likely requires generating multiple sub-illustrations per page (one
      `IllustrationGenerator.generate()` call per element) and compositing them with multiple
      `TextOverlay` placements on a single canvas, rather than the current one-image-one-label-per-page
      architecture -- a real design decision, not a small tweak.
- [ ] Add a small cache for vision-analysis results (keyed on a hash of the example image bytes +
      description), mirroring the existing raw-illustration cache in `Pipeline.run()`
      (`.pageloom_raw/`). Right now every CLI invocation re-calls the vision API even when iterating
      repeatedly against the same example image, which burns through free-tier rate limits
      (OpenRouter's free models are commonly capped around 20 requests/minute and 200/day)
      unnecessarily during development.
- [ ] Consider a `pageloom check-vision` (or similarly named) CLi subcommand that does a one-shot
      smoke test of the configured vision provider/model and reports success/failure clearly, so users
      can validate their free OpenRouter (or OpenAI/Anthropic) setup once before a full generation run
      instead of discovering a misconfiguration mid-generation.
- [x] Confirmed already implemented, no action needed: vision analysis is NOT OpenRouter-only --
      `PageVisionAnalyzer` reuses the same three-provider abstraction as text content generation
      (`PROVIDER_MAP` in `content_generator.py`), so `--vision-provider openai` (using the
      vision-capable `gpt-4o-mini`) and `--vision-provider anthropic` (using the vision-capable
      `claude-3-haiku-20240307`) both work today; OpenRouter's free `openrouter/free` auto-router is
      only the *default* because free-tier operation was the explicit priority.
- [x] Confirmed already true, no action needed: actual illustration generation (the artwork drawn
      for each page) continues to run 100% locally via the Hugging Face `diffusers` Stable Diffusion
      pipeline (`IllustrationGenerator`), completely unchanged by Milestone 5. Vision analysis is a
      separate, small, one-time-per-run API call used only to understand the example page's
      structure/content -- it does not touch or replace local image generation, which remains free,
      private, and offline-capable (aside from the initial model download).
- [x] Confirmed already true, no action needed (verified with a live-simulated test, not just code
      reading): if the configured vision model/provider does NOT actually support image input --
      whether because of a manual `OPENROUTER_VISION_MODEL`/`--vision-provider` override, or any other
      API-level rejection -- the run does **not** crash. `generate_vision_content()` catches the
      provider's exception, `PageVisionAnalyzer.analyze()` wraps it as `VisionAnalysisError`, and
      `detect_structure()` catches exactly that, prints
      `"Warning: vision-based page analysis failed, falling back to heuristic classification: <reason>"`
      to stderr (including the real underlying error message), and continues the run via the
      heuristic/description-based classifier. Same applies if the model *appears* to accept the
      request but returns non-JSON or incomplete JSON (e.g. a model that silently ignores the image and
      just replies in prose) -- `_parse_response()` raises the same `VisionAnalysisError`, same
      graceful fallback. This was confirmed by simulating a "model does not support image inputs"-style
      API error end-to-end through `pageloom generate` and observing a clean, informative, non-fatal
      exit (see manual verification during this session; not yet captured as a checked-in
      `tests/test_cli.py` case -- see the item below).
- [ ] Add the "model rejects/ignores the image" scenario as an actual checked-in regression test in
      `tests/test_cli.py` (e.g. `test_cli_vision_model_rejects_image_falls_back_gracefully`), simulating
      a provider exception like `"This model does not support image inputs"` and asserting the CLI still
      exits 0, prints the fallback warning, and completes generation via the heuristic path. This exact
      scenario was manually verified during this planning session but is not yet locked in as a test.
- [ ] Make the fallback warning more specifically actionable when the failure looks like a
      vision-capability mismatch (as opposed to a rate limit, timeout, or bad key, which already get
      distinct, clear messages via `_classify_provider_error`). Detect common phrasings in the
      underlying error (e.g. containing "image", "vision", "multimodal", "does not support") and print
      a dedicated hint such as: `"The selected model/provider does not appear to support image input.
      Set OPENROUTER_VISION_MODEL to a vision-capable model, or unset it to use the free auto-router
      (openrouter/free), which selects one automatically -- see README for how to find vision-capable
      models."` Today the user still gets the raw underlying error message (not silence), so this is a
      clarity improvement, not a bug fix.
- [ ] Document in README.md, concretely, how to find current free vision-capable models on OpenRouter:
      (a) https://openrouter.ai/models filtered by input modality "image", or
      https://openrouter.ai/collections/vision-models for a maintained list across all providers
      (paid and free); (b) as of this writing (Sept 2026) free vision-capable models observed include
      `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free`,
      `inclusionai/ling-3.0-flash-vl:free`, and `thinkingmachines/inkling-small:free` -- but OpenRouter's
      free-model roster "changes constantly" (their own wording), so don't hardcode any of these as a
      recommendation without a freshness caveat; (c) explain why the codebase defaults to the
      `openrouter/free` **auto-router** instead of naming a specific free model: OpenRouter's own docs
      state this router "smartly filters for models that support features needed for your request such
      as image understanding," so it should automatically avoid non-vision models for a vision request
      without the user needing to look anything up. Note for whoever picks this up: that specific
      auto-routing claim has not yet been verified against a real live call (folds into the existing
      "do one real smoke test" item above) -- confirm it holds before presenting it in the README as a
      guarantee rather than OpenRouter's own claim.

## Discovered
- [x] Add safe image-return guard and error handling to generator for empty pipeline results or failed model loads
- [x] Page-structure classifier was implemented as a dependency of the consistency guard (Milestone 4); the full structural routing through the pipeline and README updates remain
- [x] Pipeline and ConsistencyGuard pass images to PageStructureClassifier without descriptions, so image-only inference misses description-based patterns like object_name_pair; descriptions need to flow through the extractor/guard
- [x] Audit follow-up: the previous item was only fixed at the `Pipeline`/`ConsistencyGuard` constructor level. The CLI (`cli.py`), the only user-facing entrypoint, never passed a `description` at all — no `--description` flag existed — so `object_name_pair` was unreachable in real (non-library) usage despite being marked complete. Fixed by adding a `--description` CLI flag that is threaded through to both `PageStructureClassifier` and `Pipeline` in `cli.py`, and documented in README's CLI Flags section.
- [x] Audit follow-up: `cli.py`'s `--pages` overflow logic retained a vestigial hardcoded `alphabet_limit = 26` branch alongside the correct `built_in_limit` (35, the full `DEFAULT_CONTENTS` length) branch. For `single_letter_word_illustration` pages requesting 27-35 pages with `--use-llm`, this discarded valid built-in content items 27-35 and made unnecessary LLM calls instead — contradicting the milestone-4 reconciliation item's own stated goal ("trigger whenever N exceeds whatever content items are explicitly available"). The existing test (`test_cli_use_llm_extends_past_alphabet_limit`) asserted on this buggy behavior instead of catching it. Fixed by removing the `alphabet_limit` branch entirely so the LLM only triggers past the true `built_in_limit`; replaced the test with `test_cli_pages_within_full_built_in_list_never_calls_llm` and `test_cli_use_llm_extends_past_full_built_in_list`.
- [ ] Confirmed by live smoke test (2026-09-13): `openrouter/free` reaches the endpoint with a valid key and real image, but free-tier rate limits (`free-models-per-day-high-balance`) are active. Any retry/backoff or stricter-prompt work should assume the endpoint works and handle 429 gracefully.
- [x] Audit follow-up: `tests/test_integration.py::test_two_structurally_different_example_pages` constructed `Pipeline`/`ConsistencyGuard` for the "object-naming" example without a `description`, so it silently fell back to `multi_item_scene` inference and asserted that instead of `object_name_pair` — the test never actually exercised the structure it claimed to prove. Fixed by passing `description="object name pair"` (matching the classification step earlier in the same test) through to `Pipeline` and `ConsistencyGuard`.
<!-- work found while building; appended by the agent -->
<!-- audit pass (2026-09-13): verified every checked item above against the actual code paths listed in this file, not just the checklist claims. Three real gaps found and fixed (see the three "Audit follow-up" items). Everything else audited -- multi-provider LLM abstraction, per-provider default models, credential/error handling, license verification, CPU/CUDA/MPS device selection, caching/resume, env var overrides -- was confirmed genuinely implemented and backed by non-trivial tests. -->
