"""Grok cog package exports."""

__all__ = ["GrokCog"]


def __getattr__(name: str):
    if name == "GrokCog":
        from .cog import GrokCog

        return GrokCog
    raise AttributeError(name)
