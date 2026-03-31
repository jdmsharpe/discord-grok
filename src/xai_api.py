"""Legacy xai_api shim pointing into discord_grok."""

import warnings

from discord_grok import GrokCog

warnings.warn(
    "Importing GrokCog via xai_api is deprecated; use discord_grok.GrokCog instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GrokCog"]
