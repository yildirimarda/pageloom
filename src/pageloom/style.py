"""Style-prompt template builder for consistent illustration prompts."""

from __future__ import annotations


class StylePromptBuilder:
    """Build consistent Stable Diffusion prompts from an example page description."""

    def __init__(self, example_description: str) -> None:
        self.example_description = example_description.strip()

    def template(self) -> str:
        """Return a prompt template that embeds the example style description."""
        base = f"A children's book illustration in the style of: {self.example_description}"
        return f"{base}, content: {{letter}} {{word}}."

    def build(self, letter: str = "", word: str = "", theme: str = "") -> str:
        """Produce a concrete prompt using the example description and variables."""
        parts: list[str] = [
            "A children's book illustration",
        ]
        if theme:
            parts.append(f"theme: {theme}")
        if self.example_description:
            parts.append(f"style reference: {self.example_description}")
        if letter:
            parts.append(f"letter: {letter}")
        if word:
            parts.append(f"word: {word}")
        return ", ".join(parts) + "."
