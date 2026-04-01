import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

AUTH_PATH = Path(__file__).resolve().parents[1] / "src" / "discord_grok" / "config" / "auth.py"


def _load_auth_with_env(monkeypatch, guild_ids, collection_ids):
    monkeypatch.setenv("GUILD_IDS", guild_ids)
    monkeypatch.setenv("XAI_COLLECTION_IDS", collection_ids)
    monkeypatch.setenv("BOT_TOKEN", "dummy-token")
    monkeypatch.setenv("XAI_API_KEY", "dummy-key")

    def load_dotenv():
        return None

    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    spec = spec_from_file_location("test_auth_config", AUTH_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parses_ids_with_whitespace_and_empty_segments(monkeypatch):
    auth_config = _load_auth_with_env(
        monkeypatch,
        guild_ids=" 123 , ,456 ,   , 789 ",
        collection_ids=" alpha , ,beta ,   , gamma ",
    )

    assert auth_config.GUILD_IDS == [123, 456, 789]
    assert auth_config.XAI_COLLECTION_IDS == ["alpha", "beta", "gamma"]


def test_parses_ids_with_trailing_commas(monkeypatch):
    auth_config = _load_auth_with_env(
        monkeypatch,
        guild_ids="111,222,",
        collection_ids="collection_a,collection_b,",
    )

    assert auth_config.GUILD_IDS == [111, 222]
    assert auth_config.XAI_COLLECTION_IDS == ["collection_a", "collection_b"]


def test_validate_required_config_rejects_whitespace_only_values(monkeypatch):
    auth_config = _load_auth_with_env(monkeypatch, guild_ids="", collection_ids="")
    monkeypatch.setenv("BOT_TOKEN", "   ")
    monkeypatch.setenv("XAI_API_KEY", "\t")

    with pytest.raises(RuntimeError, match="BOT_TOKEN, XAI_API_KEY"):
        auth_config.validate_required_config()


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
    ],
)
def test_show_cost_embeds_uses_standard_boolean_parser(monkeypatch, raw_value, expected):
    monkeypatch.setenv("SHOW_COST_EMBEDS", raw_value)

    auth_config = _load_auth_with_env(monkeypatch, guild_ids="", collection_ids="")

    assert auth_config.SHOW_COST_EMBEDS is expected
