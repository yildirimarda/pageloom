"""Pluggable image-generation providers."""

from pageloom.providers.base import ImageProvider, ImageProviderCapabilities
from pageloom.providers.local_diffusers import LocalDiffusersProvider

__all__ = ["ImageProvider", "ImageProviderCapabilities", "LocalDiffusersProvider"]
