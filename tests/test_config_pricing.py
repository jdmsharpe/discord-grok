"""Tests for the YAML-backed pricing loader."""

import importlib
import sys
import textwrap
from pathlib import Path


def _reload_pricing():
    for mod_name in ("discord_grok.config.pricing",):
        sys.modules.pop(mod_name, None)
    return importlib.import_module("discord_grok.config.pricing")


class TestPricingLoader:
    def test_bundled_yaml_loads_pricing_classes(self):
        pricing = _reload_pricing()
        assert pricing.MODEL_PRICING_CLASSES["premium"] == (2.00, 0.20, 6.00)
        assert pricing.MODEL_PRICING_CLASSES["fast"] == (0.20, 0.05, 0.50)
        assert pricing.MODEL_PRICING_CLASSES["mini"] == (0.30, 0.075, 0.50)

    def test_bundled_yaml_loads_image_pricing(self):
        pricing = _reload_pricing()
        assert pricing.IMAGE_PRICING["grok-imagine-image-pro"] == 0.07
        assert pricing.IMAGE_PRICING["grok-imagine-image"] == 0.02

    def test_bundled_yaml_loads_flat_rates(self):
        pricing = _reload_pricing()
        assert pricing.VIDEO_PRICING_PER_SECOND == 0.05
        assert pricing.TTS_PRICING_PER_MILLION_CHARS == 4.20

    def test_tool_invocation_pricing(self):
        pricing = _reload_pricing()
        assert pricing.TOOL_INVOCATION_PRICING["SERVER_SIDE_TOOL_WEB_SEARCH"] == 5.00
        assert pricing.TOOL_INVOCATION_PRICING["SERVER_SIDE_TOOL_ATTACHMENT_SEARCH"] == 10.00

    def test_unknown_image_fallback(self):
        pricing = _reload_pricing()
        assert pricing.UNKNOWN_IMAGE_MODEL_PRICING == 0.07

    def test_build_model_pricing_map_still_works(self):
        """End-to-end: command_options builds per-model pricing from catalog + classes."""
        # Force pricing reload to ensure command_options picks up bundled YAML.
        _reload_pricing()
        for mod in ("discord_grok.cogs.grok.command_options",):
            sys.modules.pop(mod, None)
        from discord_grok.cogs.grok.command_options import build_model_pricing_map

        pricing_map = build_model_pricing_map()
        # grok-4.20 is in the 'premium' class.
        assert pricing_map["grok-4.20"] == (2.00, 0.20, 6.00)
        # grok-3-mini is in the 'mini' class.
        assert pricing_map["grok-3-mini"] == (0.30, 0.075, 0.50)

    def test_env_var_override_path(self, monkeypatch, tmp_path: Path):
        custom_yaml = tmp_path / "custom-pricing.yaml"
        custom_yaml.write_text(
            textwrap.dedent(
                """
                pricing_classes:
                  premium: { input_per_million: 10.0, cached_input_per_million: 1.0, output_per_million: 30.0 }
                image_generation:
                  custom-img: { per_image: 0.50 }
                video_generation:
                  per_second: 0.15
                text_to_speech:
                  per_million_chars: 9.0
                tools:
                  SERVER_SIDE_TOOL_FAKE: { per_1k_invocations: 1.0 }
                fallbacks:
                  unknown_image_model: { per_image: 0.99 }
                """
            ).strip()
        )
        monkeypatch.setenv("XAI_PRICING_PATH", str(custom_yaml))

        pricing = _reload_pricing()

        assert pricing.MODEL_PRICING_CLASSES == {"premium": (10.0, 1.0, 30.0)}
        assert pricing.IMAGE_PRICING == {"custom-img": 0.50}
        assert pricing.VIDEO_PRICING_PER_SECOND == 0.15
        assert pricing.TTS_PRICING_PER_MILLION_CHARS == 9.0
        assert pricing.UNKNOWN_IMAGE_MODEL_PRICING == 0.99
