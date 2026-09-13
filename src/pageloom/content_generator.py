"""Optional LLM-based content-list generator for arbitrary page topics."""

from __future__ import annotations

import os
from typing import Protocol


class LLMProvider(Protocol):
    """Provider-agnostic interface for LLM APIs."""

    def generate_content(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.5,
    ) -> str: ...


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
        self.model = model or os.environ.get("PAGELOOM_LLM_MODEL", "gpt-4o-mini")
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
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content
        if raw is None:
            raise RuntimeError("LLM returned empty content")
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
        self.model = model or os.environ.get("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307")

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
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text if response.content else None
        if raw is None:
            raise RuntimeError("LLM returned empty content")
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
        self.model = model or os.environ.get("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
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
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content
        if raw is None:
            raise RuntimeError("LLM returned empty content")
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
    ) -> list[str]:
        """Generate `count` content words/topics via LLM."""
        if count < 1:
            raise ValueError("count must be at least 1")
        prompt = (
            f"Generate exactly {count} simple children's book word or phrase topics "
            f"about '{theme}'. Return only a comma-separated list with no numbering, "
            f"no explanation, and no extra punctuation. Example format: Apple, Banana, Cherry"
        )
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
            fallback_words = [
                "Dragon",
                "Elephant",
                "Flower",
                "Garden",
                "House",
                "Island",
                "Journey",
                "Kingdom",
                "Light",
                "Mountain",
                "Night",
                "Ocean",
                "Planet",
                "Queen",
                "River",
                "Star",
                "Tree",
                "Universe",
                "Valley",
                "Wind",
                "Xylophone",
                "Yacht",
                "Zebra",
                "Air",
                "Ball",
                "Cloud",
                "Door",
                "Eagle",
                "Fish",
                "Gate",
            ]
            for word in fallback_words:
                if word not in result:
                    result.append(word)
                if len(result) >= count:
                    break
        return result[:count]
