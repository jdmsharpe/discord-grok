from discord import Bot, Intents

from discord_grok import GrokCog


def test_package_import_registers_cog():
    bot = Bot(intents=Intents.default())
    bot.add_cog(GrokCog(bot=bot))
    assert bot.get_cog("GrokCog") is not None
