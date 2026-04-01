import importlib
import json
import sys

import pytest


def _load_mcp_module(monkeypatch, *, inline_json=None, path_json=None, extra_env=None):
    monkeypatch.setenv("BOT_TOKEN", "dummy-token")
    monkeypatch.setenv("XAI_API_KEY", "dummy-key")

    if inline_json is None:
        monkeypatch.delenv("XAI_MCP_PRESETS_JSON", raising=False)
    else:
        monkeypatch.setenv("XAI_MCP_PRESETS_JSON", inline_json)

    if path_json is None:
        monkeypatch.delenv("XAI_MCP_PRESETS_PATH", raising=False)
    else:
        monkeypatch.setenv("XAI_MCP_PRESETS_PATH", str(path_json))

    for name, value in (extra_env or {}).items():
        monkeypatch.setenv(name, value)

    sys.modules.pop("discord_grok.config.mcp", None)
    return importlib.import_module("discord_grok.config.mcp")


def test_loads_valid_presets_from_json_and_file(monkeypatch, tmp_path):
    presets_path = tmp_path / "mcp-presets.json"
    presets_path.write_text(
        json.dumps(
            {
                "secondary": {
                    "url": "https://secondary.example.com/sse",
                    "allowed_tools": ["browse"],
                }
            }
        ),
        encoding="utf-8",
    )

    module = _load_mcp_module(
        monkeypatch,
        inline_json=json.dumps(
            {
                "primary": {
                    "url": "https://mcp.example.com/sse",
                    "allowed_tools": ["search", "run", "search"],
                }
            }
        ),
        path_json=presets_path,
    )

    assert sorted(module.XAI_MCP_PRESETS) == ["primary", "secondary"]
    assert module.XAI_MCP_PRESETS["primary"].allowed_tools == ["search", "run"]
    assert module.XAI_MCP_PRESETS["secondary"].server_url == "https://secondary.example.com/sse"


def test_rejects_duplicate_preset_names_across_sources(monkeypatch, tmp_path):
    presets_path = tmp_path / "mcp-presets.json"
    presets_path.write_text(
        json.dumps({"shared": {"url": "https://file.example.com/sse"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate xAI MCP preset names"):
        _load_mcp_module(
            monkeypatch,
            inline_json=json.dumps({"shared": {"url": "https://env.example.com/sse"}}),
            path_json=presets_path,
        )


def test_rejects_malformed_json(monkeypatch):
    with pytest.raises(ValueError, match="XAI_MCP_PRESETS_JSON must contain valid JSON"):
        _load_mcp_module(monkeypatch, inline_json="{not json}")


def test_rejects_non_object_collection(monkeypatch):
    with pytest.raises(ValueError, match="XAI_MCP_PRESETS_JSON must be a JSON object"):
        _load_mcp_module(monkeypatch, inline_json='["wrong"]')


def test_rejects_missing_url(monkeypatch):
    with pytest.raises(ValueError, match="requires a non-empty `url`"):
        _load_mcp_module(monkeypatch, inline_json=json.dumps({"broken": {}}))


def test_rejects_non_https_url(monkeypatch):
    with pytest.raises(ValueError, match="must use a valid HTTPS `url`"):
        _load_mcp_module(
            monkeypatch,
            inline_json=json.dumps({"broken": {"url": "http://example.com/sse"}}),
        )


def test_rejects_invalid_allowed_tools(monkeypatch):
    with pytest.raises(ValueError, match="`allowed_tools` must be a list of strings"):
        _load_mcp_module(
            monkeypatch,
            inline_json=json.dumps(
                {"broken": {"url": "https://example.com/sse", "allowed_tools": "search"}}
            ),
        )


def test_marks_preset_unavailable_when_authorization_env_var_missing(monkeypatch):
    module = _load_mcp_module(
        monkeypatch,
        inline_json=json.dumps(
            {
                "secure": {
                    "url": "https://secure.example.com/sse",
                    "authorization_env_var": "MISSING_TOKEN",
                }
            }
        ),
    )

    preset = module.XAI_MCP_PRESETS["secure"]
    assert preset.available is False
    assert "MISSING_TOKEN" in (preset.unavailable_reason or "")


def test_resolve_mcp_presets_reports_unknown_name(monkeypatch):
    module = _load_mcp_module(
        monkeypatch,
        inline_json=json.dumps({"known": {"url": "https://known.example.com/sse"}}),
    )

    presets, error = module.resolve_mcp_presets(["unknown"])

    assert presets == []
    assert error == "Unknown MCP preset `unknown`."


def test_build_mcp_server_config_preserves_payload_shape(monkeypatch):
    module = _load_mcp_module(
        monkeypatch,
        inline_json=json.dumps(
            {
                "trusted": {
                    "url": "https://www.mcp.example.com/sse",
                    "allowed_tools": ["search"],
                }
            }
        ),
    )

    preset = module.XAI_MCP_PRESETS["trusted"]
    config = module.build_mcp_server_config(preset)

    assert config.server_url == "https://www.mcp.example.com/sse"
    assert config.server_label == "mcp.example.com"
    assert config.allowed_tool_names == ["search"]
