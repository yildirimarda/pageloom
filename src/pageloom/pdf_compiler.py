"""Print-ready PDF compiler that combines pages into a single KDP-sized PDF."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image
from pypdf import PdfWriter

# Default square page size: 8.5 in x 8.5 in at 72 dpi = 612 x 612 pts
DEFAULT_PAGE_SIZE: tuple[float, float] = (612.0, 612.0)

# Preset page sizes in points (1 pt = 1/72 in).
PAGE_SIZE_PRESETS: dict[str, tuple[float, float]] = {
    "square": (612.0, 612.0),  # 8.5 in x 8.5 in
    "a4": (595.28, 841.89),  # 210 mm x 297 mm
    "letter": (612.0, 792.0),  # 8.5 in x 11 in
}


def parse_page_size(value: str | None) -> tuple[float, float] | None:
    """Parse a ``--page-size`` value into a (width, height) tuple in points.

    Accepts preset names (square, a4, letter) or a custom ``WIDTHxHEIGHT``
    string. Returns ``None`` when no size was requested, meaning "use the
    example image's own size".
    """
    if value is None:
        return None
    key = value.strip().lower()
    if key in PAGE_SIZE_PRESETS:
        return PAGE_SIZE_PRESETS[key]
    if "x" in key:
        width_s, height_s = key.split("x", 1)
        return (float(width_s), float(height_s))
    raise ValueError(f"Unknown page size {value!r}; use square, a4, letter or WIDTHxHEIGHT")


# Deprecated alias kept for one release. Accessing it emits a DeprecationWarning.
def __getattr__(name: str):
    if name == "KDP_PAGE_SIZE":
        import warnings

        warnings.warn(
            "KDP_PAGE_SIZE is deprecated; use DEFAULT_PAGE_SIZE instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return DEFAULT_PAGE_SIZE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class PDFCompiler:
    """Combine illustration pages into a single print-ready PDF."""

    def __init__(self, page_size: tuple[float, float] | None = None) -> None:
        self.page_size = page_size or DEFAULT_PAGE_SIZE

    def compile(
        self,
        pages: list[Image.Image] | list[str] | list[Path],
        output_path: str | Path,
    ) -> Path:
        """Compile a list of PIL images or file paths into one PDF.

        Args:
            pages: List of PIL Image objects or paths to image files.
            output_path: Destination file path for the compiled PDF.

        Returns:
            The path to the compiled PDF.
        """
        writer = PdfWriter()
        tmp_paths: list[str] = []

        try:
            for page in pages:
                if isinstance(page, (str, Path)):
                    img = Image.open(str(page))
                elif isinstance(page, Image.Image):
                    img = page
                else:
                    raise TypeError(f"Unsupported page type: {type(page)}")

                # Resize image to fill the page exactly for print-ready output
                img = img.convert("RGB")
                img = img.resize((int(self.page_size[0]), int(self.page_size[1])))

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp_paths.append(tmp_path)
                    img.save(tmp_path, "PDF", resolution=72.0)

                from pypdf import PdfReader

                reader = PdfReader(tmp_path)
                writer.append(reader)
        finally:
            for p in tmp_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer.write(str(output_path))
        return output_path
