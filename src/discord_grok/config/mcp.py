from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ..cogs.grok.models import McpServerConfig
from ..cogs.grok.tooling import MAX_MCP_LABEL_LENGTH

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class XaiMcpPreset:
    """Validated xAI MCP preset loaded from env or JSON config."""

    name: str
    server_url: str
    authorization_env_var: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    available: bool = True
    unavailable_reason: str | None = None


def _load_json_object(raw_value: str, source_name: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source_name} must contain valid JSON.") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{source_name} must be a JSON object keyed by preset name.")
    return parsed


def _validate_https_url(url: object, preset_name: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"MCP preset `{preset_name}` requires a non-empty `url`.")
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc or not parsed.hostname:
        raise ValueError(f"MCP preset `{preset_name}` must use a valid HTTPS `url`.")
    return normalized


def _validate_allowed_tools(value: object, preset_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"MCP preset `{preset_name}` `allowed_tools` must be a list of strings.")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_mcp_label(hostname: str) -> str:
    label = hostname.strip().lower()
    if label.startswith("www."):
        label = label[4:]
    if len(label) > MAX_MCP_LABEL_LENGTH:
        label = label[:MAX_MCP_LABEL_LENGTH]
    return label or "mcp"


def _validate_preset(name: str, raw_value: object) -> XaiMcpPreset:
    if not isinstance(raw_value, dict):
        raise ValueError(f"MCP preset `{name}` must be an object.")

    supported_keys = {"url", "authorization_env_var", "allowed_tools"}
    extra_keys = sorted(set(raw_value) - supported_keys)
    if extra_keys:
        raise ValueError(
            f"MCP preset `{name}` contains unsupported keys: {', '.join(extra_keys)}."
        )

    authorization_env_var = raw_value.get("authorization_env_var")
    if authorization_env_var is not None and not isinstance(authorization_env_var, str):
        raise ValueError(f"MCP preset `{name}` `authorization_env_var` must be a string.")

    preset = XaiMcpPreset(
        name=name,
        server_url=_validate_https_url(raw_value.get("url"), name),
        authorization_env_var=authorization_env_var,
        allowed_tools=_validate_allowed_tools(raw_value.get("allowed_tools"), name),
    )

    if preset.authorization_env_var and not os.getenv(preset.authorization_env_var):
        LOGGER.warning(
            "xAI MCP preset `%s` is unavailable because `%s` is not set.",
            name,
            preset.authorization_env_var,
        )
        return XaiMcpPreset(
            name=preset.name,
            server_url=preset.server_url,
            authorization_env_var=preset.authorization_env_var,
            allowed_tools=preset.allowed_tools,
            available=False,
            unavailable_reason=(
                f"MCP preset `{name}` requires the `{preset.authorization_env_var}` env var."
            ),
        )

    return preset


def load_xai_mcp_presets() -> dict[str, XaiMcpPreset]:
    """Load xAI MCP presets from JSON env text and/or a JSON file path."""
    merged: dict[str, object] = {}

    inline_json = os.getenv("XAI_MCP_PRESETS_JSON", "").strip()
    if inline_json:
        merged.update(_load_json_object(inline_json, "XAI_MCP_PRESETS_JSON"))

    presets_path = os.getenv("XAI_MCP_PRESETS_PATH", "").strip()
    if presets_path:
        file_data = Path(presets_path).read_text(encoding="utf-8")
        path_presets = _load_json_object(file_data, "XAI_MCP_PRESETS_PATH")
        duplicate_names = sorted(set(merged) & set(path_presets))
        if duplicate_names:
            raise ValueError(
                "Duplicate xAI MCP preset names found across env and file config: "
                + ", ".join(duplicate_names)
            )
        merged.update(path_presets)

    presets: dict[str, XaiMcpPreset] = {}
    for name, raw_value in merged.items():
        presets[name] = _validate_preset(name, raw_value)
    return presets


XAI_MCP_PRESETS = load_xai_mcp_presets()


def parse_mcp_preset_names(raw_value: str | None) -> list[str]:
    """Parse a comma-separated list of preset names from Discord command input."""
    if raw_value is None:
        return []
    parsed_names: list[str] = []
    seen: set[str] = set()
    for piece in raw_value.split(","):
        name = piece.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parsed_names.append(name)
    return parsed_names


def resolve_mcp_presets(
    preset_names: list[str],
) -> tuple[list[XaiMcpPreset], str | None]:
    """Resolve preset names to validated xAI MCP presets."""
    presets: list[XaiMcpPreset] = []
    for name in preset_names:
        preset = XAI_MCP_PRESETS.get(name)
        if preset is None:
            return [], f"Unknown MCP preset `{name}`."
        if not preset.available:
            return [], preset.unavailable_reason or f"MCP preset `{name}` is unavailable."
        presets.append(preset)
    return presets, None


def build_mcp_server_config(preset: XaiMcpPreset) -> McpServerConfig:
    """Convert a resolved xAI MCP preset into the chat-layer MCP config."""
    parsed = urlparse(preset.server_url)
    hostname = parsed.hostname or "mcp"
    return McpServerConfig(
        server_url=preset.server_url,
        server_label=_normalize_mcp_label(hostname),
        allowed_tool_names=list(preset.allowed_tools),
    )


__all__ = [
    "XAI_MCP_PRESETS",
    "XaiMcpPreset",
    "build_mcp_server_config",
    "load_xai_mcp_presets",
    "parse_mcp_preset_names",
    "resolve_mcp_presets",
]
