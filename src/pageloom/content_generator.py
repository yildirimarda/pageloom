"""Optional LLM-based content-list generator for arbitrary page topics."""

from __future__ import annotations

import os


class LLMContentGenerator:
    """Generate arbitrary content words/topics using an LLM API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("PAGELOOM_LLM_API_KEY")
        self.model = model or os.environ.get("PAGELOOM_LLM_MODEL", "gpt-4o-mini")
        self.endpoint = endpoint or os.environ.get("PAGELOOM_LLM_ENDPOINT")

    def generate(
        self,
        count: int,
        theme: str = "children's book alphabet words",
    ) -> list[str]:
        """Generate `count` content words/topics via LLM."""
        if count < 1:
            raise ValueError("count must be at least 1")
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
        prompt = (
            f"Generate exactly {count} simple children's book word or phrase topics "
            f"about '{theme}'. Return only a comma-separated list with no numbering, "
            f"no explanation, and no extra punctuation. Example format: Apple, Banana, Cherry"
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=max(100, count * 20),
            )
            raw = response.choices[0].message.content
            if raw is None:
                raise RuntimeError("LLM returned empty content")
            items = [item.strip().strip('"') for item in raw.split(",") if item.strip()]
            # Filter out non-string items just in case
            result: list[str] = []
            for item in items:
                if isinstance(item, str) and item:
                    result.append(item)
                    if len(result) >= count:
                        break
            if len(result) < count:
                # Pad with generic fallback words so we always return `count` items
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
        except Exception as exc:
            raise RuntimeError(f"LLM content generation failed: {exc}") from exc
