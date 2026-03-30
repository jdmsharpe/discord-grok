"""Legacy auth shim pointing into discord_grok.config."""

import warnings

from discord_grok.config.auth import *  # noqa: F401,F403

warnings.warn(
    "Importing config.auth via src/config is deprecated; use discord_grok.config.auth instead.",
    DeprecationWarning,
    stacklevel=2,
)
