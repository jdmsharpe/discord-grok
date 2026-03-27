import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GUILD_IDS = [int(id) for id in os.getenv("GUILD_IDS", "").split(",") if id]
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_COLLECTION_IDS = [
    collection_id
    for collection_id in os.getenv("XAI_COLLECTION_IDS", "").split(",")
    if collection_id
]
SHOW_COST_EMBEDS = os.getenv("SHOW_COST_EMBEDS", "true").lower() in ("true", "1", "yes")
