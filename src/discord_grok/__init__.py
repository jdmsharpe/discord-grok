"""Public namespace for the Discord Grok bot."""

from .cogs.grok.cog import xAIAPI  # noqa: F401
from .config.auth import BOT_TOKEN  # noqa: F401

__all__ = ["xAIAPI", "BOT_TOKEN"]
