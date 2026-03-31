from discord import Bot, Intents

from discord_grok import GrokCog


def test_namespaced_import_registers_cog():
    intents = Intents.default()
    intents.presences = False
    intents.members = True
    intents.message_content = True
    intents.guilds = True
    bot = Bot(intents=intents)
    bot.add_cog(GrokCog(bot=bot))
    assert bot.get_cog("GrokCog") is not None
