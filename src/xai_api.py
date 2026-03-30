"""Legacy xai_api shim pointing into discord_grok."""

import warnings

from discord_grok import xAIAPI

warnings.warn(
    "Importing xAIAPI via xai_api is deprecated; use discord_grok.xAIAPI instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["xAIAPI"]
