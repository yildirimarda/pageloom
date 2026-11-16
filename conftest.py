"""Shared test fixtures.

Mocks OpenRouter's models endpoint so that ``settings.get_openrouter_vision_model()``
(which now discovers a vision-capable model from the endpoint instead of defaulting
to ``openrouter/free``) is deterministic and offline in the test suite. Only requests
to ``https://openrouter.ai/api/v1/models`` are intercepted; every other URL falls
through to the real ``urllib.request.urlopen``.
"""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import patch

import pytest

_MODELS_URL = "https://openrouter.ai/api/v1/models"

_FAKE_VISION_MODELS_PAYLOAD = {
    "data": [
        {
            "id": "google/gemma-4-26b-a4b-it:free",
            "architecture": {"modality": "text+image->text"},
        },
        {"id": "openai/gpt-4o-mini", "architecture": {"modality": "text->text"}},
    ]
}


class _FakeUrlResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(_FAKE_VISION_MODELS_PAYLOAD).encode("utf-8")


def _urlopen_side_effect(req, *args, **kwargs):
    url = req.full_url if hasattr(req, "full_url") else str(req)
    if _MODELS_URL in url:
        return _FakeUrlResponse()
    return urllib.request.urlopen(req, *args, **kwargs)


@pytest.fixture(autouse=True)
def _mock_openrouter_models_endpoint():
    with patch(
        "pageloom.providers.openrouter.urllib.request.urlopen", side_effect=_urlopen_side_effect
    ):
        yield
