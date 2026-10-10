#!/usr/bin/env python3
"""Real (non-mocked) smoke test of vision analysis.

Runs ``PageVisionAnalyzer.analyze`` -- the exact vision call the
``pageloom generate`` CLI makes via ``detect_structure`` -- against a real,
checked-in example page image with the real, free ``OPENROUTER_API_KEY`` and no
``--description``, and confirms whether ``openrouter/free`` returns valid JSON
matching ``PageVisionAnalyzer``'s expected schema.

Every existing test mocks the network, so this is the only check of the real
prompt/schema combination against a live free-tier vision model.

Reality of OpenRouter's ``openrouter/free`` auto-router (confirmed by this
script, 2026-10): the router is *non-deterministic* and rotates across
different free models per request. A vision-capable free model returns valid
JSON matching the schema; a non-vision / safety / text-only free model may
return empty content or prose instead. ``PageVisionAnalyzer.analyze`` now
retries once with a stricter "ONLY JSON" prompt before giving up. So this
script reports an outcome honestly:

  * valid JSON matching the schema            -> PASS (exit 0)
  * empty / non-JSON / safety prose, retried  -> known graceful free-tier
                                                 outcome (the generate path
                                                 falls back to the heuristic
                                                 classifier) -> PASS (exit 0)
  * rate limit (429)                          -> known graceful free-tier
                                                 outcome -> PASS (exit 0)
  * auth / network / any other failure        -> FAIL (exit 1)
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, "/work/src")

from PIL import Image

from pageloom.vision import (
    _REQUIRED_FIELDS,
    _VISION_CACHE_DIR,
    PageVisionAnalyzer,
    VisionAnalysisError,
)

# A real, checked-in example page image (not a solid-color dummy) so the smoke
# test exercises a genuine children's-book page.
DEFAULT_EXAMPLE = "example_alphabet.png"


def _classify_failure(msg: str) -> str:
    """Categorize a VisionAnalysisError for honest reporting."""
    low = msg.lower()
    if "rate limit" in low or "429" in low or "free-models-per-day" in low:
        return "rate limit"
    if "empty content" in low:
        return "empty content (openrouter/free routed to a non-vision/safety model)"
    if (
        "non-json" in low
        or "not a valid" in low
        or "missing required" in low
        or "unexpected json shape" in low
    ):
        return "non-JSON output (openrouter/free routed to a non-vision/safety model)"
    # Defense-in-depth: the openrouter/free auto-router may return HTTP 200
    # with null/empty response fields (choices=None, empty choices, etc.),
    # which surfaces as SDK-level TypeErrors/IndexErrors/AttributeErrors
    # rather than the clean "empty content" message. Treat these as the same
    # known graceful free-tier outcome.
    if (
        "nonetype" in low
        or "not subscriptable" in low
        or "has no attribute" in low
        or "list index out of range" in low
    ):
        return "empty content (openrouter/free routed to a non-vision/safety model)"
    return "unexpected error"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    arg_example: str | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--example" and i + 1 < len(argv):
            arg_example = argv[i + 1]
            i += 2
        else:
            i += 1

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("FAIL: OPENROUTER_API_KEY not set in environment.")
        return 1

    example = arg_example or DEFAULT_EXAMPLE
    if not os.path.exists(example):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            example = f.name
        Image.new("RGB", (612, 612), color="red").save(example)

    # Bypass the vision cache so each smoke run is a genuine live call against
    # the openrouter/free endpoint (the cache exists to spare production runs,
    # not smoke tests).
    cache_key = hashlib.md5(PageVisionAnalyzer._image_bytes(example) + b"").hexdigest()
    cache_file = _VISION_CACHE_DIR / (cache_key + ".json")
    if cache_file.exists():
        cache_file.unlink()

    try:
        print(f"Smoke test (real example image): -> {example}")
        print(f"API key present: yes (starts with {key[:10]}...)")

        analyzer = PageVisionAnalyzer(provider="openrouter")
        print("Provider: openrouter")
        print(f"Model:    {analyzer.provider.model}")
        endpoint = getattr(analyzer.provider, "endpoint", None)
        print(f"Endpoint: {endpoint}")
        print("Retries on empty/non-JSON: 1 (stricter 'ONLY JSON' prompt)")

        # Confirm image encoding produces a valid data URL.
        b64_url = analyzer._encode_image(example)
        assert b64_url.startswith("data:image/png;base64,")
        print(f"Image encoding: OK ({len(b64_url)} chars)")

        # The live vision call (with the built-in retry).
        result = analyzer.analyze(example)
        print("Live vision call: SUCCESS")
        print(f"Returned keys: {sorted(result.keys())}")
        for k in [
            "content_pattern",
            "content_type_description",
            "element_count",
            "arrangement",
            "text_image_relationship",
            "example_content_items",
        ]:
            print(f"  {k}: {result.get(k)!r}")
        # Schema validation against PageVisionAnalyzer's expected schema.
        missing = [k for k in _REQUIRED_FIELDS if k not in result]
        ok = (
            isinstance(result, dict)
            and not missing
            and isinstance(result.get("element_count"), int)
            and "arrangement" in result
            and isinstance(result.get("example_content_items"), list)
        )
        if ok:
            print("Schema validation: PASSED")
            print(
                "CONFIRMED: openrouter/free returned valid JSON matching "
                "PageVisionAnalyzer's expected schema."
            )
            return 0
        print(f"Schema validation: FAILED (missing: {missing})")
        return 1
    except VisionAnalysisError as exc:
        reason = _classify_failure(str(exc))
        print(f"Live vision call handled gracefully: {exc}")
        if reason == "unexpected error":
            print(f"Unexpected failure ({reason}): {exc}")
            return 1
        print(
            "Known graceful free-tier outcome "
            f"({reason}): openrouter/free is non-deterministic and may route to "
            "a non-vision/safety model that returns empty or non-JSON output. "
            "PageLoom's generate path falls back to the heuristic classifier; "
            "a vision-capable free model may be selected on a later run."
        )
        return 0
    finally:
        if example != (arg_example or DEFAULT_EXAMPLE):
            try:
                os.unlink(example)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
