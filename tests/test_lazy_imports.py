import importlib
import sys
from contextlib import contextmanager


@contextmanager
def _fresh_package(prefix: str):
    original_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == prefix or name.startswith(f"{prefix}.")
    }

    for name in original_modules:
        sys.modules.pop(name, None)

    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_top_level_package_import_is_lazy_for_grok_cog():
    with _fresh_package("discord_grok"):
        package = importlib.import_module("discord_grok")

        assert "discord_grok.cogs.grok.cog" not in sys.modules
        assert "GrokCog" in package.__all__


def test_top_level_bot_token_export_does_not_import_grok_cog():
    with _fresh_package("discord_grok"):
        package = importlib.import_module("discord_grok")

        _ = package.BOT_TOKEN

        assert "discord_grok.cogs.grok.cog" not in sys.modules


def test_cog_package_import_is_lazy():
    with _fresh_package("discord_grok"):
        package = importlib.import_module("discord_grok.cogs.grok")

        assert "discord_grok.cogs.grok.cog" not in sys.modules
        assert "GrokCog" in package.__all__
