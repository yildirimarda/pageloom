"""License verification for Stable Diffusion models used in commercial books."""

from __future__ import annotations

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
    """Verify whether a Stable Diffusion model's license permits commercial use."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5") -> None:
        self.model_id = model_id

    def verify(
        self, intended_use: str = "commercial", context: str = "book intended for sale"
    ) -> dict:
        """Check the model license and return verification details.

        Args:
            intended_use: "commercial" or other use type.
            context: Human-readable description of the use (e.g. book intended for sale).

        Returns:
            A dict with verification result, license info, and notes.

        Raises:
            LicenseVerificationError: If the license does not permit the intended use.
        """
        info = COMMERCIAL_USE_MODELS.get(self.model_id)
        if info is not None:
            permitted_str = info.get("commercial_use_permitted", "").lower()
            permitted = permitted_str in ("true", "yes", "1")
            result = {
                "verified": permitted,
                "model_id": self.model_id,
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

        # For unknown models, require explicit confirmation for safety.
        result = {
            "verified": False,
            "model_id": self.model_id,
            "license": "Unknown / Not in verified list",
            "license_url": None,
            "commercial_use_permitted": False,
            "notes": "This model is not in the verified commercial-use list. Verify the license manually before using generated images in a book intended for sale.",
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
    context: str = "book intended for sale",
) -> dict:
    """Convenience function to verify a model for commercial book generation."""
    verifier = LicenseVerifier(model_id=model_id)
    return verifier.verify(intended_use="commercial", context=context)
