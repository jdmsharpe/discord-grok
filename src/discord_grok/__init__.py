"""Public namespace for the Discord Grok bot."""

__all__ = ["GrokCog", "BOT_TOKEN"]


def __getattr__(name: str):
    if name == "GrokCog":
        from .cogs.grok import GrokCog

        return GrokCog
    if name == "BOT_TOKEN":
        from .config.auth import BOT_TOKEN

        return BOT_TOKEN
    raise AttributeError(name)
