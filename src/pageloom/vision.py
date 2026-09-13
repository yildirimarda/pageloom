"""Vision-based page-structure analyzer.

Looks directly at an example page image with a vision-capable LLM and infers
an open-ended structure description (content pattern, structural attributes,
and example content items actually visible on the page) -- without requiring
the user to hand-write a description. This replaces the closed 3-category
heuristic in `page_structure_classifier.py` as the primary detection path,
so arbitrary page types (numbers, digits, word-image matching, shapes,
colors, etc.) are supported, not just the three originally hardcoded
patterns.

Defaults to OpenRouter's free auto-router model (see
`settings.get_vision_provider`/`settings.get_openrouter_vision_model`) so the
default, zero-config experience costs nothing. OpenAI and Anthropic are also
supported for users who prefer them.

Note: the Stable Diffusion model used by `IllustrationGenerator` (via the
`diffusers` library) is a text-to-image *generation* model only -- it cannot
look at an existing image and describe it, so it cannot be reused for this
purpose. Analyzing an input image requires a genuinely multimodal
(vision-capable) model, which is what this module calls out to.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from pageloom import settings
from pageloom.content_generator import PROVIDER_MAP, LLMProvider

_ANALYSIS_PROMPT = """You are analyzing a single page from a children's educational book so that more pages matching its structure can be generated later.

Look at the image and respond with ONLY a JSON object (no markdown fences, no commentary) with exactly these keys:
{
  "content_pattern": "<short snake_case id for this page type, e.g. number_count_matching, digit_illustration, word_image_matching, single_letter_word_illustration>",
  "content_type_description": "<one sentence in plain English describing what this page type is and what content item pairs with the illustration(s)>",
  "element_count": <integer, how many distinct illustrated elements/items appear on the page>,
  "arrangement": "<one word: centered, grid, row, column, or scattered>",
  "text_image_relationship": "<one of: label_per_image, multiple_labels, no_text>",
  "example_content_items": ["<the actual word/number/label(s) visible on THIS page>", "..."]
}

Base "example_content_items" strictly on what is actually visible/legible on the page (e.g. if the page shows the number "3" next to three ducks, an example item could be "3 ducks"). If you cannot read any text, infer a reasonable label from the illustration instead. IMPORTANT: respond with ONLY the JSON object — no extra text, no explanation, no markdown fences, even if you are rate-limited or interrupted."""

_REQUIRED_FIELDS = [
    "content_pattern",
    "content_type_description",
    "element_count",
    "arrangement",
    "text_image_relationship",
]

_ENV_VARS_BY_PROVIDER = {
    "openai": ["OPENAI_API_KEY", "PAGELOOM_LLM_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY", "PAGELOOM_LLM_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY", "PAGELOOM_LLM_API_KEY"],
}


_VISION_CACHE_DIR = Path(".pageloom_vision")


class VisionAnalysisError(RuntimeError):
    """Raised when vision-based page-structure analysis is unavailable or fails.

    Callers (Pipeline, ConsistencyGuard, the CLI) catch this specifically so
    they can fall back to the heuristic PageStructureClassifier instead of
    crashing the whole run -- vision analysis is a best-effort upgrade, not a
    hard requirement.
    """


class PageVisionAnalyzer:
    """Analyze an example page image with a vision-capable LLM.

    Returns an open-ended structure dict shaped compatibly with
    `PageStructureClassifier.classify()`'s output (same core keys), plus two
    extra keys -- `content_type_description` and `example_content_items` --
    that let `LLMContentGenerator` generate matching content for ANY page
    type, not just the three legacy hardcoded patterns.
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.provider_name = provider or settings.get_vision_provider()
        provider_cls = PROVIDER_MAP.get(self.provider_name)
        if provider_cls is None:
            raise ValueError(
                f"Unknown vision provider '{self.provider_name}'. "
                f"Supported: {', '.join(PROVIDER_MAP)}"
            )
        default_models = {
            "openai": settings.get_openai_vision_model(),
            "anthropic": settings.get_anthropic_vision_model(),
            "openrouter": settings.get_openrouter_vision_model(),
        }
        init_kwargs: dict = {
            "api_key": api_key,
            "model": model or default_models.get(self.provider_name),
            "endpoint": endpoint,
        }
        if self.provider_name == "anthropic":
            init_kwargs.pop("endpoint", None)
        self.provider: LLMProvider = provider_cls(**init_kwargs)
        provider_key = getattr(self.provider, "api_key", None)
        if not provider_key:
            env_vars = _ENV_VARS_BY_PROVIDER.get(self.provider_name, ["PAGELOOM_LLM_API_KEY"])
            raise VisionAnalysisError(
                f"Vision provider '{self.provider_name}' selected but API key is missing. "
                f"Set the appropriate environment variable ({', '.join(env_vars)}) or pass "
                f"api_key explicitly. (Tip: OpenRouter's free tier needs only an "
                f"OPENROUTER_API_KEY -- see README for setup.)"
            )

    @staticmethod
    def _encode_image(source: str | Path | Image.Image) -> str:
        image = source if isinstance(source, Image.Image) else Image.open(str(source))
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _image_bytes(source: str | Path | Image.Image) -> bytes:
        image = source if isinstance(source, Image.Image) else Image.open(str(source))
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _cache_key(self, source: str | Path | Image.Image, description: str) -> str:
        raw_bytes = self._image_bytes(source)
        key_data = raw_bytes + description.encode("utf-8")
        return hashlib.md5(key_data).hexdigest()

    def _cache_path_for(self, source: str | Path | Image.Image, description: str) -> Path:
        _VISION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return _VISION_CACHE_DIR / (self._cache_key(source, description) + ".json")

    def analyze(self, source: str | Path | Image.Image, description: str = "") -> dict:
        """Send the example page image to the vision model and return a structure dict.

        Raises VisionAnalysisError on any failure (network, rate limit,
        malformed response) rather than silently returning a guess -- callers
        decide whether/how to fall back.

        Results are cached under `.pageloom_vision/`, keyed on a hash of the
        example image bytes plus the description string, so repeated runs
        against the same image skip the vision API call.
        """
        cache_path = self._cache_path_for(source, description)
        try:
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError, KeyError):
            # If the cache file is corrupt or unreadable, ignore it and
            # fall through to a fresh analysis.
            pass

        try:
            image_data_url = self._encode_image(source)
        except Exception as exc:
            raise VisionAnalysisError(
                f"Could not read example image for vision analysis: {exc}"
            ) from exc

        prompt = _ANALYSIS_PROMPT
        if description:
            prompt += (
                f"\n\nThe page creator describes it as: {description!r}. Use this as a "
                f"hint but trust what you actually see in the image."
            )
        try:
            raw = self.provider.generate_vision_content(
                prompt=prompt,
                image_data_url=image_data_url,
                max_tokens=400,
                temperature=0.2,
            )
        except Exception as exc:
            raise VisionAnalysisError(f"Vision-based page analysis failed: {exc}") from exc
        result = self._parse_response(raw)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f)
        except OSError:
            # Cache write failure is non-fatal; the analysis result is
            # still valid.
            pass
        return result

    @staticmethod
    def _parse_response(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    raise VisionAnalysisError(
                        f"Vision model returned non-JSON output that could not be "
                        f"parsed: {raw[:200]!r}"
                    ) from exc
            else:
                raise VisionAnalysisError(
                    f"Vision model returned non-JSON output that could not be parsed: {raw[:200]!r}"
                ) from exc

        if not isinstance(data, dict):
            raise VisionAnalysisError(
                f"Vision model returned unexpected JSON shape (expected an object): {raw[:200]!r}"
            )

        missing = [key for key in _REQUIRED_FIELDS if key not in data]
        if missing:
            raise VisionAnalysisError(
                f"Vision model response is missing required field(s): {', '.join(missing)}. "
                f"Raw response: {raw[:200]!r}"
            )

        data.setdefault("example_content_items", [])
        if not isinstance(data.get("example_content_items"), list):
            data["example_content_items"] = []
        try:
            data["element_count"] = int(data["element_count"])
        except (TypeError, ValueError):
            data["element_count"] = 1
        return data
