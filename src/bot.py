"""
Notes
---------------
May need to use this command to install pycord
python -m pip install --upgrade --no-deps --force-reinstall git+https://github.com/Pycord-Development/pycord
"""

from discord import Bot, Intents

from xai_api import xAIAPI
from config.auth import BOT_TOKEN

if __name__ == "__main__":
    intents = Intents.default()
    intents.presences = False
    intents.members = True
    intents.message_content = True
    intents.guilds = True
    bot = Bot(intents=intents)
    bot.add_cog(xAIAPI(bot=bot))
    bot.run(BOT_TOKEN)
