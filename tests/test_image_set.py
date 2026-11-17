"""Unit tests for the plain image-set renderer (pageloom.image_set)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pageloom.image_set import ImageSetRenderer, load_contents, round_to_multiple
from pageloom.providers.base import ImageProviderCapabilities


def _make_provider(generate_side_effect=None, supports_references=False, cost=1.0):
    provider = MagicMock()
    provider.capabilities = ImageProviderCapabilities(
        max_reference_images=3 if supports_references else 0
    )
    if generate_side_effect is not None:
        provider.generate.side_effect = generate_side_effect
    else:
        provider.generate.return_value = Image.new("RGB", (64, 64), color="blue")
    provider.estimated_cost_usd.return_value = cost
    return provider


def test_load_contents_positional_only():
    assert load_contents(["a", "b"], None) == ["a", "b"]


def test_load_contents_skips_blank_positional():
    assert load_contents(["a", "", "   ", "b"], None) == ["a", "b"]


def test_load_contents_txt_file(tmp_path):
    file = tmp_path / "prompts.txt"
    file.write_text("cat\n\n# comment\ndog\n", encoding="utf-8")
    assert load_contents([], file) == ["cat", "dog"]


def test_load_contents_json_strings(tmp_path):
    file = tmp_path / "prompts.json"
    file.write_text(json.dumps(["apple", "banana"]), encoding="utf-8")
    assert load_contents([], file) == ["apple", "banana"]


def test_load_contents_json_objects(tmp_path):
    file = tmp_path / "prompts.json"
    file.write_text(json.dumps([{"prompt": "cat"}, {"prompt": "dog"}]), encoding="utf-8")
    assert load_contents([], file) == ["cat", "dog"]


def test_load_contents_merges_positional_and_file(tmp_path):
    file = tmp_path / "prompts.txt"
    file.write_text("from-file\n", encoding="utf-8")
    assert load_contents(["first"], file) == ["first", "from-file"]


def test_load_contents_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_contents([], Path("does-not-exist.txt"))


def test_load_contents_bad_json_raises(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_contents([], file)


def test_load_contents_bad_json_names_line(tmp_path):
    """Invalid JSON names the offending line in the error message."""
    file = tmp_path / "bad.json"
    file.write_text('[\n  "apple",\n  not valid\n]', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        load_contents([], file)


def test_load_contents_bad_entry_names_line(tmp_path):
    """A non-string, non-prompt JSON entry names its line in the error."""
    file = tmp_path / "bad.json"
    file.write_text('["apple", 42, "banana"]', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_contents([], file)


@pytest.mark.parametrize(
    "dim,expected",
    [(618, 624), (800, 800), (513, 520), (512, 512), (1, 8), (7, 8)],
)
def test_round_to_multiple(dim, expected):
    assert round_to_multiple(dim) == expected


def test_renderer_render_calls_provider_per_prompt(tmp_path):
    prompts = ["a red ball", "a blue cube"]
    with patch("pageloom.image_set.build_image_provider") as mock_build:
        provider = _make_provider()
        mock_build.return_value = provider
        renderer = ImageSetRenderer(prompts, width=64, height=64, cache_dir=tmp_path / "cache")
        images = renderer.render()
    assert len(images) == 2
    assert provider.generate.call_count == 2
    calls = [call.kwargs["prompt"] for call in provider.generate.call_args_list]
    assert calls == prompts


def test_renderer_style_reference_forwarded_as_reference(tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), color="green").save(ref)
    prompts = ["a prompt"]
    with patch("pageloom.image_set.build_image_provider") as mock_build:
        provider = _make_provider(supports_references=True)
        mock_build.return_value = provider
        renderer = ImageSetRenderer(prompts, style_reference=str(ref), width=64, height=64)
        renderer.render()
    assert provider.generate.call_count == 1
    kwargs = provider.generate.call_args.kwargs
    assert "references" in kwargs
    assert len(kwargs["references"]) == 1
    assert isinstance(kwargs["references"][0], Image.Image)


def test_renderer_no_references_when_provider_lacks_support(tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), color="green").save(ref)
    prompts = ["a prompt"]
    with patch("pageloom.image_set.build_image_provider") as mock_build:
        provider = _make_provider(supports_references=False)
        mock_build.return_value = provider
        renderer = ImageSetRenderer(prompts, style_reference=str(ref), width=64, height=64)
        renderer.render()
    kwargs = provider.generate.call_args.kwargs
    assert kwargs.get("references") is None


def test_renderer_uses_passed_provider_directly(tmp_path):
    prompts = ["a prompt"]
    provider = _make_provider()
    renderer = ImageSetRenderer(prompts, provider=provider, width=64, height=64)
    renderer.render()
    provider.generate.assert_called_once()


def test_renderer_estimate_cost_without_calling_provider(tmp_path):
    prompts = ["a", "b", "c"]
    with patch("pageloom.image_set.build_image_provider") as mock_build:
        provider = _make_provider(cost=2.5)
        mock_build.return_value = provider
        renderer = ImageSetRenderer(prompts, width=64, height=64)
        assert renderer.estimated_total_cost_usd() == 7.5
        provider.generate.assert_not_called()


def test_renderer_caches_and_force_regenerates(tmp_path):
    cache_dir = tmp_path / "cache"
    prompts = ["a prompt"]
    with patch("pageloom.image_set.build_image_provider") as mock_build:
        provider = _make_provider()
        mock_build.return_value = provider
        renderer = ImageSetRenderer(prompts, width=64, height=64, cache_dir=cache_dir)
        renderer.render()
        assert provider.generate.call_count == 1
        # Second render reuses the cache.
        renderer.render()
        assert provider.generate.call_count == 1
        # --force regenerates.
        renderer.force = True
        renderer.render()
        assert provider.generate.call_count == 2


def test_renderer_budget_stops_generation(tmp_path):
    prompts = ["a", "b", "c"]
    with patch("pageloom.image_set.build_image_provider") as mock_build:
        provider = _make_provider(cost=10.0)
        mock_build.return_value = provider
        renderer = ImageSetRenderer(prompts, width=64, height=64, max_cost_usd=15.0)
        images = renderer.render()
    # First image costs 10 (total 10 <= 15), second would bring total to 20 > 15 -> stop.
    assert len(images) == 1
    assert provider.generate.call_count == 1
    assert renderer.budget_stopped is True


def test_renderer_odd_dimensions_rounded_up(tmp_path):
    prompts = ["a prompt"]
    with patch("pageloom.image_set.build_image_provider") as mock_build:
        provider = _make_provider()
        mock_build.return_value = provider
        renderer = ImageSetRenderer(prompts, width=618, height=800)
        renderer.render()
    kwargs = provider.generate.call_args.kwargs
    assert kwargs["width"] % 8 == 0
    assert kwargs["height"] % 8 == 0
    assert kwargs["width"] >= 618
    assert kwargs["height"] >= 800
