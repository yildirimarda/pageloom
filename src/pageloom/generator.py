"""Stable Diffusion illustration generator using Hugging Face diffusers.

The real implementation lives in
:class:`pageloom.providers.local_diffusers.LocalDiffusersProvider`; this module
keeps ``IllustrationGenerator`` importable as a thin alias so existing callers
and tests keep working.
"""

from __future__ import annotations

import torch  # noqa: F401 -- re-exported so tests can patch pageloom.generator.torch.cuda.is_available
from diffusers import (
    StableDiffusionPipeline,  # noqa: F401 -- re-exported so tests can patch pageloom.generator.StableDiffusionPipeline.from_pretrained
)

from pageloom.providers.local_diffusers import LocalDiffusersProvider


class IllustrationGenerator(LocalDiffusersProvider):
    """Thin alias for :class:`LocalDiffusersProvider`.

    Generate illustrations using a Stable Diffusion pipeline.

    A trivial subclass (rather than a plain assignment) so existing code that
    subclasses ``IllustrationGenerator`` keeps working, and so
    ``pageloom.pipeline`` can keep patching ``pageloom.pipeline.IllustrationGenerator``.
    """
