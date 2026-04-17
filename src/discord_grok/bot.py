from discord import Bot, Intents

from .cogs.grok.cog import GrokCog
from .config.auth import BOT_TOKEN, validate_required_config
from .logging_setup import configure_logging


def main() -> None:
    validate_required_config()
    configure_logging()
    intents = Intents.default()
    intents.presences = False
    intents.members = True
    intents.message_content = True
    intents.guilds = True
    bot = Bot(intents=intents)
    bot.add_cog(GrokCog(bot=bot))
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
