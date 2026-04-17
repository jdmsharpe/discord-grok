"""Load xAI Grok pricing from pricing.yaml.

Chat pricing is organized by pricing class (premium / fast / mini / etc.); the
model → class mapping lives in ``command_options.py`` alongside the catalog.
This loader owns only the class → price mapping plus the flat pricing for
image / video / TTS and per-tool server-side invocation rates.

Override the bundled file at runtime by setting ``XAI_PRICING_PATH`` to a
different YAML path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _resolve_pricing_path() -> Path:
    override = os.getenv("XAI_PRICING_PATH")
    if override:
        return Path(override)
    return Path(__file__).with_name("pricing.yaml")


def _load_raw() -> dict[str, Any]:
    path = _resolve_pricing_path()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a YAML mapping at the top level.")
    return data


_RAW: dict[str, Any] = _load_raw()
_CLASSES: dict[str, dict[str, Any]] = _RAW.get("pricing_classes") or {}
_IMAGE: dict[str, dict[str, Any]] = _RAW.get("image_generation") or {}
_VIDEO: dict[str, Any] = _RAW.get("video_generation") or {}
_TTS: dict[str, Any] = _RAW.get("text_to_speech") or {}
_TOOLS: dict[str, dict[str, Any]] = _RAW.get("tools") or {}
_FALLBACKS: dict[str, dict[str, Any]] = _RAW.get("fallbacks") or {}

MODEL_PRICING_CLASSES: dict[str, tuple[float, float, float]] = {
    class_id: (
        float(cfg["input_per_million"]),
        float(cfg["cached_input_per_million"]),
        float(cfg["output_per_million"]),
    )
    for class_id, cfg in _CLASSES.items()
}

IMAGE_PRICING: dict[str, float] = {
    model_id: float(cfg["per_image"]) for model_id, cfg in _IMAGE.items()
}

VIDEO_PRICING_PER_SECOND: float = float(_VIDEO.get("per_second", 0.05))

TTS_PRICING_PER_MILLION_CHARS: float = float(_TTS.get("per_million_chars", 4.20))

TOOL_INVOCATION_PRICING: dict[str, float] = {
    tool_id: float(cfg["per_1k_invocations"]) for tool_id, cfg in _TOOLS.items()
}

UNKNOWN_IMAGE_MODEL_PRICING: float = float(
    (_FALLBACKS.get("unknown_image_model") or {}).get("per_image", 0.07)
)


__all__ = [
    "IMAGE_PRICING",
    "MODEL_PRICING_CLASSES",
    "TOOL_INVOCATION_PRICING",
    "TTS_PRICING_PER_MILLION_CHARS",
    "UNKNOWN_IMAGE_MODEL_PRICING",
    "VIDEO_PRICING_PER_SECOND",
]
