"""Public namespace for the Discord Grok bot."""

from .cogs.grok.cog import GrokCog  # noqa: F401
from .config.auth import BOT_TOKEN  # noqa: F401

__all__ = ["GrokCog", "BOT_TOKEN"]
