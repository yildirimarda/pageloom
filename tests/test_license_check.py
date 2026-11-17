"""Tests for license verification before commercial book generation."""

import pytest

from pageloom.license_check import (
    COMMERCIAL_USE_MODELS,
    DEFAULT_CONTEXT,
    LicenseVerificationError,
    LicenseVerifier,
    verify_model_for_commercial_use,
)


def test_known_commercial_model_verifies():
    verifier = LicenseVerifier("runwayml/stable-diffusion-v1-5")
    result = verifier.verify(intended_use="commercial", context="book intended for sale")
    assert result["verified"] is True
    assert result["commercial_use_permitted"] is True
    assert "CreativeML OpenRAIL-M" in result["license"]


def test_known_commercial_model_raises_on_non_commercial_intention():
    # Even for known commercial models, if intended_use is not commercial,
    # it should not raise (it just verifies the info is available).
    verifier = LicenseVerifier("runwayml/stable-diffusion-v1-5")
    result = verifier.verify(intended_use="research", context="academic study")
    assert result["verified"] is True


def test_unknown_model_raises_for_commercial_use():
    verifier = LicenseVerifier("unknown/model-id-123")
    with pytest.raises(LicenseVerificationError) as exc_info:
        verifier.verify(intended_use="commercial", context="book intended for sale")
    assert "Unknown model" in str(exc_info.value)
    assert "book intended for sale" in str(exc_info.value)


def test_convenience_function_verifies_default_model():
    result = verify_model_for_commercial_use()
    assert result["verified"] is True
    assert result["model_id"] == "runwayml/stable-diffusion-v1-5"


def test_convenience_function_raises_for_unknown_model():
    with pytest.raises(LicenseVerificationError) as exc_info:
        verify_model_for_commercial_use(model_id="unknown/model-id")
    assert "Unknown model" in str(exc_info.value)


def test_known_models_dict_contains_openrail_m():
    for info in COMMERCIAL_USE_MODELS.values():
        assert info.get("license") is not None
        assert info.get("license_url") is not None


def test_default_context_is_neutral():
    verifier = LicenseVerifier("runwayml/stable-diffusion-v1-5")
    result = verifier.verify(intended_use="commercial")
    assert result["context"] == DEFAULT_CONTEXT
    assert result["context"] != "book intended for sale"


def test_local_provider_checks_model_license_as_today():
    verifier = LicenseVerifier("runwayml/stable-diffusion-v1-5", provider="local")
    result = verifier.verify(intended_use="commercial")
    assert result["verified"] is True
    assert result["provider"] == "local"
    assert "CreativeML OpenRAIL-M" in result["license"]


def test_online_provider_returns_tos_note_and_does_not_raise():
    # Even an unknown model id is not blocked for an online provider.
    verifier = LicenseVerifier("unknown/model-id-123", provider="openai")
    result = verifier.verify(intended_use="commercial")
    assert result["verified"] is True
    assert result["provider"] == "openai"
    assert result["license"] is None
    assert result["commercial_use_permitted"] is None
    assert "terms of service" in result["notes"]
    assert "output-ownership" in result["notes"]


def test_online_provider_known_model_still_governed_by_tos():
    # A known commercial model id does not trigger the local license check
    # when the provider is online.
    verifier = LicenseVerifier("runwayml/stable-diffusion-v1-5", provider="gemini")
    result = verifier.verify(intended_use="commercial")
    assert result["verified"] is True
    assert result["provider"] == "gemini"
    assert result["license"] is None
    assert "terms of service" in result["notes"]
