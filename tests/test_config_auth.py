import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

AUTH_PATH = Path(__file__).resolve().parents[1] / "src" / "discord_grok" / "config" / "auth.py"


def _load_auth_with_env(monkeypatch, guild_ids, collection_ids):
    monkeypatch.setenv("GUILD_IDS", guild_ids)
    monkeypatch.setenv("XAI_COLLECTION_IDS", collection_ids)

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
