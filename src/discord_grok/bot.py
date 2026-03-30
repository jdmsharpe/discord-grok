import logging

from discord import Bot, Intents

from .cogs.grok.cog import xAIAPI
from .config.auth import BOT_TOKEN

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    intents = Intents.default()
    intents.presences = False
    intents.members = True
    intents.message_content = True
    intents.guilds = True
    bot = Bot(intents=intents)
    bot.add_cog(xAIAPI(bot=bot))
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
