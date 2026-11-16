"""Tests for the Black Forest Labs FLUX.2 image provider.

Covers the three request shapes (BFL's own API, fal, Replicate),
multi-reference input, model/host resolution from settings, credential
errors, and the reference-count limit. All provider calls are mocked so
the suite runs offline and deterministically.
"""

import base64
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from PIL import Image

from pageloom import settings
from pageloom.providers import (
    IMAGE_PROVIDER_MAP,
    BFLImageProvider,
    ImageProvider,
    build_image_provider,
)

_VALID_B64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
    "DwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _image_bytes() -> bytes:
    return base64.b64decode(_VALID_B64_PNG)


@contextmanager
def _mock_urlopen(responses):
    """Mock ``urllib.request.urlopen`` returning canned responses.

    ``responses`` maps a URL (or URL prefix) to a dict with optional
    ``status``, ``body`` (dict, serialized as JSON) and ``raw`` (bytes).
    A ``read()`` returns JSON bytes for dict bodies or raw bytes
    otherwise. The object supports the context-manager protocol.
    """

    class _Response:
        def __init__(self, body, raw=None):
            self._body = body
            self._raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            if self._raw is not None:
                return self._raw
            return json.dumps(self._body).encode("utf-8")

    def _opener(req, *args, **kwargs):
        url = getattr(req, "full_url", None) or str(req)
        data = getattr(req, "data", None)
        method = getattr(req, "method", "GET")
        headers = dict(getattr(req, "header_items", list)())
        body = json.loads(data.decode("utf-8")) if data else None
        for prefix, spec in responses.items():
            if url.startswith(prefix):
                return _Response(spec.get("body", {}), spec.get("raw"))
        raise AssertionError(
            f"No mocked response for URL: {url} (method={method}, body={body}, headers={headers})"
        )

    import pageloom.providers.bfl as bfl_module

    original = bfl_module.urllib.request.urlopen
    bfl_module.urllib.request.urlopen = _opener
    try:
        yield _opener
    finally:
        bfl_module.urllib.request.urlopen = original


@contextmanager
def _mock_sleep():
    import pageloom.providers.bfl as bfl_module

    original = bfl_module.time.sleep
    bfl_module.time.sleep = lambda *args, **kwargs: None
    try:
        yield
    finally:
        bfl_module.time.sleep = original


# -- interface / capabilities --


def test_bfl_image_provider_is_an_image_provider():
    provider = BFLImageProvider(api_key="test-key")
    assert isinstance(provider, ImageProvider)
    assert provider.capabilities.max_reference_images == 3
    assert provider.capabilities.supports_edit is True
    assert provider.capabilities.supports_negative_prompt is False


def test_bfl_image_provider_model_id_read_from_settings(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_MODEL", "flux-dev")
    provider = BFLImageProvider(api_key="test-key")
    assert provider.model_id == "flux-dev"


def test_bfl_image_provider_model_id_defaults_to_settings_default(monkeypatch):
    monkeypatch.delenv("PAGELOOM_FLUX_MODEL", raising=False)
    provider = BFLImageProvider(api_key="test-key")
    assert provider.model_id == settings.get_flux_model()


def test_bfl_image_provider_host_defaults_to_bfl(monkeypatch):
    monkeypatch.delenv("PAGELOOM_FLUX_HOST", raising=False)
    provider = BFLImageProvider(api_key="test-key")
    assert provider.host == "bfl"


def test_bfl_image_provider_host_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_HOST", "replicate")
    provider = BFLImageProvider(api_key="test-key")
    assert provider.host == "replicate"


# -- credential resolution --


def test_bfl_image_provider_bfl_api_key_from_env(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "bfl-key")
    provider = BFLImageProvider()
    assert provider.api_key == "bfl-key"


def test_bfl_image_provider_fal_api_key_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_HOST", "fal")
    monkeypatch.setenv("FAL_KEY", "fal-key")
    provider = BFLImageProvider()
    assert provider.api_key == "fal-key"


def test_bfl_image_provider_replicate_api_key_from_env(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_HOST", "replicate")
    monkeypatch.setenv("REPLICATE_API_KEY", "rep-key")
    provider = BFLImageProvider()
    assert provider.api_key == "rep-key"


def test_bfl_image_provider_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "env-key")
    provider = BFLImageProvider(api_key="explicit-key")
    assert provider.api_key == "explicit-key"


# -- BFL's own API request shape --


def test_bfl_generate_bfl_host_sends_launch_and_polls(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "bfl-key")
    responses = {
        "https://api.bfl.ai/v1/flux-pro-1.1": {
            "body": {"id": "req-123"},
        },
        "https://api.bfl.ai/v1/get_result": {
            "body": {
                "status": "Ready",
                "result": {"sample": ["https://bfl.example.com/out.png"]},
            }
        },
        "https://bfl.example.com/out.png": {"raw": _image_bytes()},
    }
    with _mock_urlopen(responses), _mock_sleep():
        provider = BFLImageProvider(api_key="bfl-key")
        image = provider.generate("a red apple", width=1024, height=1024)
    assert isinstance(image, Image.Image)
    assert image.size == (1, 1)
    assert image.mode == "RGB"


def test_bfl_generate_bfl_host_request_body(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "bfl-key")
    captured = {}

    def _opener(req, *args, **kwargs):
        url = getattr(req, "full_url", None) or str(req)
        if url.startswith("https://api.bfl.ai/v1/flux-pro-1.1"):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["headers"] = dict(req.header_items())
            return _FakeResp({"id": "req-1"})
        if url.startswith("https://api.bfl.ai/v1/get_result"):
            return _FakeResp(
                {
                    "status": "Ready",
                    "result": {"sample": ["https://bfl.example.com/out.png"]},
                }
            )
        if url.startswith("https://bfl.example.com/"):
            return _FakeResp(None, raw=_image_bytes())
        raise AssertionError(f"Unexpected URL: {url}")

    class _FakeResp:
        def __init__(self, body, raw=None):
            self._body = body
            self._raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            if self._raw is not None:
                return self._raw
            return json.dumps(self._body).encode("utf-8")

    import pageloom.providers.bfl as bfl_module

    original = bfl_module.urllib.request.urlopen
    bfl_module.urllib.request.urlopen = _opener
    try:
        with _mock_sleep():
            provider = BFLImageProvider(api_key="bfl-key")
            provider.generate("match the example", width=1024, height=1024, seed=42)
    finally:
        bfl_module.urllib.request.urlopen = original

    body = captured["body"]
    assert body["prompt"] == "match the example"
    assert body["width"] == 1024
    assert body["height"] == 1024
    assert body["seed"] == 42
    assert body["output_format"] == "png"
    assert "image_prompt" not in body
    headers = captured["headers"]
    assert headers.get("Authorization") == "Bearer bfl-key"
    assert {k.lower(): v for k, v in headers.items()}.get("content-type") == "application/json"


def test_bfl_generate_bfl_host_with_multi_reference(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "bfl-key")
    captured = {}

    def _opener(req, *args, **kwargs):
        url = getattr(req, "full_url", None) or str(req)
        if url.startswith("https://api.bfl.ai/v1/flux-pro-1.1"):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp({"id": "req-1"})
        if url.startswith("https://api.bfl.ai/v1/get_result"):
            return _FakeResp(
                {
                    "status": "Ready",
                    "result": {"sample": ["https://bfl.example.com/out.png"]},
                }
            )
        if url.startswith("https://bfl.example.com/"):
            return _FakeResp(None, raw=_image_bytes())
        raise AssertionError(f"Unexpected URL: {url}")

    class _FakeResp:
        def __init__(self, body, raw=None):
            self._body = body
            self._raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            if self._raw is not None:
                return self._raw
            return json.dumps(self._body).encode("utf-8")

    import pageloom.providers.bfl as bfl_module

    original = bfl_module.urllib.request.urlopen
    bfl_module.urllib.request.urlopen = _opener
    try:
        with _mock_sleep():
            provider = BFLImageProvider(api_key="bfl-key")
            ref1 = Image.new("RGB", (1024, 1024), color="red")
            ref2 = Image.new("RGB", (1024, 1024), color="green")
            provider.generate(
                "match both examples",
                width=1024,
                height=1024,
                references=[ref1, ref2],
            )
    finally:
        bfl_module.urllib.request.urlopen = original

    prompts = captured["body"]["image_prompt"]
    assert len(prompts) == 2
    assert prompts[0]["strength"] == 0.8
    assert prompts[1]["strength"] == 0.8
    assert prompts[0]["image"].startswith("data:image/png;base64,")
    assert prompts[1]["image"].startswith("data:image/png;base64,")
    assert prompts[0]["image"] != prompts[1]["image"]


# -- fal request shape --


def test_bfl_generate_fal_host_request_body(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_HOST", "fal")
    monkeypatch.setenv("FAL_KEY", "fal-key")
    captured = {}

    def _opener(req, *args, **kwargs):
        url = getattr(req, "full_url", None) or str(req)
        if url.startswith("https://fal.run/fal-ai/flux-pro-1.1"):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["headers"] = dict(req.header_items())
            return _FakeResp({"image": {"url": "https://fal.example.com/out.png"}})
        if url.startswith("https://fal.example.com/"):
            return _FakeResp(None, raw=_image_bytes())
        raise AssertionError(f"Unexpected URL: {url}")

    class _FakeResp:
        def __init__(self, body, raw=None):
            self._body = body
            self._raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            if self._raw is not None:
                return self._raw
            return json.dumps(self._body).encode("utf-8")

    import pageloom.providers.bfl as bfl_module

    original = bfl_module.urllib.request.urlopen
    bfl_module.urllib.request.urlopen = _opener
    try:
        provider = BFLImageProvider(api_key="fal-key")
        ref = Image.new("RGB", (1024, 1024), color="blue")
        provider.generate("a cat", width=1024, height=1024, references=[ref], seed=7)
    finally:
        bfl_module.urllib.request.urlopen = original

    body = captured["body"]
    assert body["prompt"] == "a cat"
    assert body["image_urls"][0].startswith("data:image/png;base64,")
    assert body["aspect_ratio"] == "1024:1024"
    assert body["seed"] == 7
    headers = captured["headers"]
    assert headers.get("Authorization") == "Key fal-key"


# -- Replicate request shape --


def test_bfl_generate_replicate_host_request_body(monkeypatch):
    monkeypatch.setenv("PAGELOOM_FLUX_HOST", "replicate")
    monkeypatch.setenv("REPLICATE_API_KEY", "rep-key")
    captured = {}

    def _opener(req, *args, **kwargs):
        url = getattr(req, "full_url", None) or str(req)
        if url.startswith(
            "https://api.replicate.com/v1/models/black-forest-labs/flux-pro-1.1/predictions"
        ):
            captured["launch_body"] = json.loads(req.data.decode("utf-8"))
            captured["launch_headers"] = dict(req.header_items())
            return _FakeResp({"id": "pred-1"})
        if url.startswith("https://api.replicate.com/v1/predictions/pred-1"):
            return _FakeResp(
                {
                    "status": "succeeded",
                    "output": ["https://replicate.example.com/out.png"],
                }
            )
        if url.startswith("https://replicate.example.com/"):
            return _FakeResp(None, raw=_image_bytes())
        raise AssertionError(f"Unexpected URL: {url}")

    class _FakeResp:
        def __init__(self, body, raw=None):
            self._body = body
            self._raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            if self._raw is not None:
                return self._raw
            return json.dumps(self._body).encode("utf-8")

    import pageloom.providers.bfl as bfl_module

    original = bfl_module.urllib.request.urlopen
    bfl_module.urllib.request.urlopen = _opener
    try:
        with _mock_sleep():
            provider = BFLImageProvider(api_key="rep-key")
            provider.generate("a dog", width=1024, height=1024)
    finally:
        bfl_module.urllib.request.urlopen = original

    body = captured["launch_body"]
    assert body["input"]["prompt"] == "a dog"
    assert body["input"]["aspect_ratio"] == "1024:1024"
    assert "seed" not in body["input"]
    headers = captured["launch_headers"]
    assert headers.get("Authorization") == "Token rep-key"


# -- reference-count limit --


def test_bfl_generate_rejects_too_many_references():
    provider = BFLImageProvider(api_key="test-key")
    refs = [Image.new("RGB", (64, 64)) for _ in range(4)]
    with pytest.raises(ValueError, match="at most 3 reference images"):
        provider.generate("a prompt", references=refs)


# -- build_image_provider registration --


def test_build_image_provider_flux_registered():
    assert "flux" in IMAGE_PROVIDER_MAP
    assert IMAGE_PROVIDER_MAP["flux"] is BFLImageProvider


def test_build_image_provider_flux_missing_credential_raises_clear_error(monkeypatch):
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="BFL_API_KEY|FAL_KEY|REPLICATE_API_KEY"),
    ):
        build_image_provider(provider="flux")


def test_build_image_provider_flux_with_credential(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "bfl-key")
    with patch("pageloom.providers.bfl.urllib.request.urlopen"):
        provider = build_image_provider(provider="flux")
    assert isinstance(provider, BFLImageProvider)
    assert provider.host == "bfl"
