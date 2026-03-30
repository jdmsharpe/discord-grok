import warnings

from discord import Bot, Intents

from discord_grok import xAIAPI


def test_namespaced_import_registers_cog():
    intents = Intents.default()
    intents.presences = False
    intents.members = True
    intents.message_content = True
    intents.guilds = True
    bot = Bot(intents=intents)
    bot.add_cog(xAIAPI(bot=bot))
    assert bot.get_cog("xAIAPI") is not None


def test_shim_import_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from xai_api import xAIAPI as shim_class  # noqa: F401

        assert shim_class is xAIAPI
    assert any(item.category is DeprecationWarning for item in caught)
