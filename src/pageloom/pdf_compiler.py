"""Print-ready PDF compiler that combines pages into a single KDP-sized PDF."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image
from pypdf import PdfWriter

# Standard Amazon KDP children's book square size: 8.5 in x 8.5 in at 72 dpi = 612 x 612 pts
KDP_PAGE_SIZE: tuple[float, float] = (612.0, 612.0)


class PDFCompiler:
    """Combine illustration pages into a single print-ready KDP PDF."""

    def __init__(self, page_size: tuple[float, float] | None = None) -> None:
        self.page_size = page_size or KDP_PAGE_SIZE

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
