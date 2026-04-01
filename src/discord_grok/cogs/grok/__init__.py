"""Grok cog package exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cog import GrokCog

__all__ = ["GrokCog"]


def __getattr__(name: str):
    if name == "GrokCog":
        from .cog import GrokCog

        return GrokCog
    raise AttributeError(name)
