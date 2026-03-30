"""Legacy config shim (temporarily points into discord_grok)."""

import warnings

from discord_grok.config.auth import (
    BOT_TOKEN,
    GUILD_IDS,
    SHOW_COST_EMBEDS,
    XAI_API_KEY,
    XAI_COLLECTION_IDS,
)

warnings.warn(
    "Importing config from the top-level package is deprecated; use discord_grok.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BOT_TOKEN",
    "GUILD_IDS",
    "SHOW_COST_EMBEDS",
    "XAI_API_KEY",
    "XAI_COLLECTION_IDS",
]
