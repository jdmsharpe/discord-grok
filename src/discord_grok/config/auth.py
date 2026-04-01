import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GUILD_IDS = [
    int(guild_id)
    for guild_id in (token.strip() for token in os.getenv("GUILD_IDS", "").split(","))
    if guild_id
]
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_COLLECTION_IDS = [
    collection_id.strip()
    for collection_id in os.getenv("XAI_COLLECTION_IDS", "").split(",")
    if collection_id
    if collection_id.strip()
]
SHOW_COST_EMBEDS = os.getenv("SHOW_COST_EMBEDS", "true").lower() in ("true", "1", "yes")
