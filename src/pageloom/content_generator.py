"""Optional LLM-based content-list generator for arbitrary page topics."""

from __future__ import annotations

import os
from typing import Protocol

from pageloom import settings


class LLMProvider(Protocol):
    """Provider-agnostic interface for LLM APIs."""

    def generate_content(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.5,
    ) -> str: ...

    def generate_vision_content(
        self,
        prompt: str,
        image_data_url: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> str: ...


def _classify_provider_error(exc: Exception, provider_name: str) -> str:
    """Map a raw SDK exception to a clear, provider-labeled error message."""
    exc_name = exc.__class__.__name__
    if "RateLimit" in exc_name:
        return f"LLM provider '{provider_name}' rate limit exceeded."
    if "Timeout" in exc_name or "APITimeout" in exc_name:
        return f"LLM provider '{provider_name}' request timed out."
    if "Authentication" in exc_name or "APIKey" in exc_name or "InvalidKey" in exc_name:
        return f"LLM provider '{provider_name}' invalid or missing API key."
    return f"LLM provider '{provider_name}' API call failed: {exc}"


def _parse_data_url(data_url: str) -> tuple[str, str, str]:
    """Split a `data:<media_type>;base64,<data>` URL into (media_type, encoding, data)."""
    if not data_url.startswith("data:"):
        raise ValueError("Expected a data: URL")
    header, _, b64_data = data_url.partition(",")
    media_part = header[len("data:") :]
    media_type, _, encoding = media_part.partition(";")
    return media_type or "image/png", encoding or "base64", b64_data


class OpenAIProvider:
    """OpenAI-compatible provider (OpenAI, OpenRouter)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("PAGELOOM_LLM_API_KEY")
        )
        self.model = model or settings.get_openai_llm_model()
        self.endpoint = endpoint or os.environ.get("PAGELOOM_LLM_ENDPOINT")

    def generate_content(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.5,
    ) -> str:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "LLM content generation requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        client = openai.OpenAI(
            api_key=self.api_key or "sk-dummy",
            base_url=self.endpoint,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(_classify_provider_error(exc, self.__class__.__name__)) from exc
        raw = response.choices[0].message.content
        if raw is None:
            raise RuntimeError("LLM returned empty content")
        return raw

    def generate_vision_content(
        self,
        prompt: str,
        image_data_url: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "Vision-based page analysis requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        client = openai.OpenAI(
            api_key=self.api_key or "sk-dummy",
            base_url=self.endpoint,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(_classify_provider_error(exc, self.__class__.__name__)) from exc
        raw = response.choices[0].message.content
        if raw is None:
            raise RuntimeError("Vision model returned empty content")
        return raw


class AnthropicProvider:
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("PAGELOOM_LLM_API_KEY")
        )
        self.model = model or settings.get_anthropic_llm_model()

    def generate_content(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.5,
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "LLM content generation with Anthropic requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from exc
        client = anthropic.Anthropic(api_key=self.api_key or "sk-dummy")
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise RuntimeError(_classify_provider_error(exc, self.__class__.__name__)) from exc
        raw = response.content[0].text if response.content else None
        if raw is None:
            raise RuntimeError("LLM returned empty content")
        return raw

    def generate_vision_content(
        self,
        prompt: str,
        image_data_url: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Vision-based page analysis with Anthropic requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from exc
        media_type, _, b64_data = _parse_data_url(image_data_url)
        client = anthropic.Anthropic(api_key=self.api_key or "sk-dummy")
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise RuntimeError(_classify_provider_error(exc, self.__class__.__name__)) from exc
        raw = response.content[0].text if response.content else None
        if raw is None:
            raise RuntimeError("Vision model returned empty content")
        return raw


class OpenRouterProvider:
    """OpenRouter provider (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("PAGELOOM_LLM_API_KEY")
        )
        self.model = model or settings.get_openrouter_llm_model()
        # Default OpenRouter endpoint if not specified
        default_endpoint = endpoint or os.environ.get("PAGELOOM_LLM_ENDPOINT")
        self.endpoint = default_endpoint or "https://openrouter.ai/api/v1"

    def generate_content(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.5,
    ) -> str:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "LLM content generation requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        client = openai.OpenAI(
            api_key=self.api_key or "sk-dummy",
            base_url=self.endpoint,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(_classify_provider_error(exc, self.__class__.__name__)) from exc
        raw = response.choices[0].message.content
        if raw is None:
            raise RuntimeError("LLM returned empty content")
        return raw

    def generate_vision_content(
        self,
        prompt: str,
        image_data_url: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "Vision-based page analysis requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        client = openai.OpenAI(
            api_key=self.api_key or "sk-dummy",
            base_url=self.endpoint,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(_classify_provider_error(exc, self.__class__.__name__)) from exc
        raw = response.choices[0].message.content
        if raw is None:
            raise RuntimeError("Vision model returned empty content")
        return raw


PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openrouter": OpenRouterProvider,
}


class LLMContentGenerator:
    """Generate arbitrary content words/topics using an LLM API."""

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.provider_name = provider or os.environ.get("PAGELOOM_LLM_PROVIDER", "openai")
        provider_cls = PROVIDER_MAP.get(self.provider_name)
        if provider_cls is None:
            raise ValueError(
                f"Unknown LLM provider '{self.provider_name}'. Supported: {', '.join(PROVIDER_MAP)}"
            )
        init_kwargs: dict = {"api_key": api_key, "model": model, "endpoint": endpoint}
        if self.provider_name == "anthropic":
            init_kwargs.pop("endpoint", None)
        self.provider: LLMProvider = provider_cls(**init_kwargs)
        # Validate that the selected provider's API key is present
        provider_key = getattr(self.provider, "api_key", None)
        if not provider_key:
            env_vars = {
                "openai": ["OPENAI_API_KEY", "PAGELOOM_LLM_API_KEY"],
                "anthropic": ["ANTHROPIC_API_KEY", "PAGELOOM_LLM_API_KEY"],
                "openrouter": ["OPENROUTER_API_KEY", "PAGELOOM_LLM_API_KEY"],
            }.get(self.provider_name, ["PAGELOOM_LLM_API_KEY"])
            raise RuntimeError(
                f"LLM provider '{self.provider_name}' selected but API key is missing. "
                f"Set the appropriate environment variable ({', '.join(env_vars)}) or pass api_key explicitly."
            )

    @property
    def api_key(self) -> str | None:
        return getattr(self.provider, "api_key", None)

    @property
    def model(self) -> str | None:
        return getattr(self.provider, "model", None)

    @property
    def endpoint(self) -> str | None:
        return getattr(self.provider, "endpoint", None)

    def generate(
        self,
        count: int,
        theme: str = "children's book alphabet words",
        page_structure: dict | None = None,
    ) -> list[str]:
        """Generate `count` content words/topics via LLM, routed through detected page structure."""
        if count < 1:
            raise ValueError("count must be at least 1")

        # Route through detected page structure so generated content matches the page type
        content_pattern = (page_structure or {}).get(
            "content_pattern", "single_letter_word_illustration"
        )
        element_count = (page_structure or {}).get("element_count", 1)
        arrangement = (page_structure or {}).get("arrangement", "centered")
        content_type_description = (page_structure or {}).get("content_type_description")
        example_content_items = (page_structure or {}).get("example_content_items") or []

        if content_type_description:
            # Open-ended path: the page structure came from PageVisionAnalyzer (or any
            # caller that supplies a free-text description of the page type), so this
            # is NOT limited to the three legacy content_pattern buckets below -- it
            # works for any page type (numbers, digits, word-image matching, shapes,
            # colors, etc.) described in plain language.
            examples_str = (
                ", ".join(example_content_items[:5])
                if example_content_items
                else "items appropriate for this page type"
            )
            prompt_template = (
                f"Generate exactly {count} new content items for a children's book page. "
                f"The page type is: {content_type_description} "
                f"(roughly {element_count} illustrated element(s) per page, arranged in a {arrangement} layout). "
                f"Each item should be a short label/topic in the same style and spirit as these examples "
                f"from the original page, but NOT repeating them: {examples_str}. "
                f"Return only a comma-separated list with no numbering, no explanation, and no extra punctuation."
            )
        elif content_pattern == "object_name_pair":
            prompt_template = (
                f"Generate exactly {count} simple children's book object names (single objects like Cat, Dog, Tree). "
                f"Each item should be a single object name suitable for a naming page with a photo-realistic illustration and its name label. "
                f"Return only a comma-separated list with no numbering, no explanation, and no extra punctuation. "
                f"Example format: Cat, Dog, Tree"
            )
        elif content_pattern == "multi_item_scene":
            prompt_template = (
                f"Generate exactly {count} themed scene descriptions for a children's book page with multiple labeled items "
                f"(e.g., {element_count} items arranged in a {arrangement} layout). Each item should describe a group or scene theme. "
                f"Return only a comma-separated list with no numbering, no explanation, and no extra punctuation. "
                f"Example format: Picnic in the Park, Garden Party, Forest Adventure"
            )
        elif content_pattern == "single_letter_word_illustration":
            prompt_template = (
                f"Generate exactly {count} simple children's book word or phrase topics about '{theme}'. "
                f"Return only a comma-separated list with no numbering, no explanation, and no extra punctuation. "
                f"Example format: Apple, Banana, Cherry"
            )
        else:
            # Fallback to generic letter/word style for unknown patterns
            prompt_template = (
                f"Generate exactly {count} simple children's book word or phrase topics about '{theme}'. "
                f"Return only a comma-separated list with no numbering, no explanation, and no extra punctuation. "
                f"Example format: Apple, Banana, Cherry"
            )

        prompt = prompt_template
        try:
            raw = self.provider.generate_content(
                prompt=prompt,
                max_tokens=max(100, count * 20),
                temperature=0.5,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM content generation failed: {exc}") from exc
        if raw is None:
            raise RuntimeError("LLM returned empty content")
        items = [item.strip().strip('"') for item in raw.split(",") if item.strip()]
        result: list[str] = []
        for item in items:
            if isinstance(item, str) and item:
                result.append(item)
                if len(result) >= count:
                    break
        if len(result) < count:
            raise RuntimeError(
                f"LLM provider '{self.provider_name}' returned insufficient content: "
                f"expected {count}, got {len(result)}. The API call may have failed "
                f"(timeout, rate limit, invalid key) or returned truncated output. "
                f"Check your credentials and network connection."
            )
        return result[:count]
