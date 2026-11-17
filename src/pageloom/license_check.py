"""License verification for Stable Diffusion models used in commercial books."""

from __future__ import annotations

# Neutral default context used when no specific use is described. Replaces the
# previously hard-coded "book intended for sale" so the check is not tied to a
# single use case.
DEFAULT_CONTEXT = "commercial use"

# Known model IDs and their licenses that permit commercial use.
# OpenRAIL-M allows commercial use with attribution and same-license modifications.
COMMERCIAL_USE_MODELS: dict[str, dict[str, str]] = {
    "runwayml/stable-diffusion-v1-5": {
        "license": "CreativeML OpenRAIL-M",
        "license_url": "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        "commercial_use_permitted": "True",
        "notes": "Commercial use permitted with attribution; modifications must be shared under the same license.",
    },
    "sd-legacy/stable-diffusion-v1-5": {
        "license": "CreativeML OpenRAIL-M",
        "license_url": "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        "commercial_use_permitted": "True",
        "notes": "Commercial use permitted with attribution; modifications must be shared under the same license.",
    },
}

# Licenses that explicitly DO NOT permit commercial use for generated images in books intended for sale.
NON_COMMERCIAL_LICENSES: set[str] = {
    "CreativeML OpenRAIL (non-M)",
    "BigScience BLOOM OpenRAIL (non-M, research only)",
    "Research Only",
    "Non-Commercial",
    "Academic Use Only",
}


class LicenseVerificationError(Exception):
    """Raised when the model's license does not permit commercial use."""


class LicenseVerifier:
    """Verify whether a model's license permits the intended use.

    Local (self-hosted) models are checked against the verified commercial-use
    list. Online providers are not checked against a local model license: their
    generated images are governed by the provider's terms of service and
    output-ownership rules instead.
    """

    def __init__(
        self, model_id: str = "runwayml/stable-diffusion-v1-5", provider: str = "local"
    ) -> None:
        self.model_id = model_id
        self.provider = provider

    def verify(self, intended_use: str = "commercial", context: str = DEFAULT_CONTEXT) -> dict:
        """Check the model license and return verification details.

        Args:
            intended_use: "commercial" or other use type.
            context: Human-readable description of the use (e.g. commercial use).

        Returns:
            A dict with verification result, license info, and notes.

        Raises:
            LicenseVerificationError: If the license does not permit the intended use.
        """
        if self.provider != "local":
            # Online providers are not checked against a local model license:
            # generated images are governed by the provider's terms of service
            # and output-ownership rules instead.
            return {
                "verified": True,
                "model_id": self.model_id,
                "provider": self.provider,
                "license": None,
                "license_url": None,
                "commercial_use_permitted": None,
                "notes": (
                    f"Online provider '{self.provider}': generated images are governed by the "
                    "provider's terms of service and output-ownership rules, not by a local "
                    "model license. Review the provider's terms before using generated images "
                    "in a commercial book."
                ),
                "intended_use": intended_use,
                "context": context,
            }

        info = COMMERCIAL_USE_MODELS.get(self.model_id)
        if info is not None:
            permitted_str = info.get("commercial_use_permitted", "").lower()
            permitted = permitted_str in ("true", "yes", "1")
            result = {
                "verified": permitted,
                "model_id": self.model_id,
                "provider": self.provider,
                "license": info.get("license"),
                "license_url": info.get("license_url"),
                "commercial_use_permitted": permitted,
                "notes": info.get("notes", ""),
                "intended_use": intended_use,
                "context": context,
            }
            if intended_use == "commercial" and not permitted:
                raise LicenseVerificationError(
                    f"Model '{self.model_id}' license ({info.get('license')}) does not permit "
                    f"{context}. Aborting generation for safety."
                )
            return result

        # For unknown local models, require explicit confirmation for safety.
        result = {
            "verified": False,
            "model_id": self.model_id,
            "provider": self.provider,
            "license": "Unknown / Not in verified list",
            "license_url": None,
            "commercial_use_permitted": False,
            "notes": (
                f"This model is not in the verified commercial-use list. Verify the license "
                f"manually before using generated images in a {context}."
            ),
            "intended_use": intended_use,
            "context": context,
        }
        if intended_use == "commercial":
            raise LicenseVerificationError(
                f"Unknown model '{self.model_id}' — cannot verify commercial-use permission. "
                f"Please confirm the model's license permits {context} before proceeding."
            )
        return result


def verify_model_for_commercial_use(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    context: str = DEFAULT_CONTEXT,
    provider: str = "local",
) -> dict:
    """Convenience function to verify a model for commercial book generation."""
    verifier = LicenseVerifier(model_id=model_id, provider=provider)
    return verifier.verify(intended_use="commercial", context=context)
