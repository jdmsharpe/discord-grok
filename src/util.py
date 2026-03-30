"""Legacy util shim pointing into discord_grok.cogs.grok.tooling."""

import warnings

from discord_grok.cogs.grok.tooling import *  # noqa: F401,F403

warnings.warn(
    "Importing util is deprecated; use discord_grok.cogs.grok.tooling instead.",
    DeprecationWarning,
    stacklevel=2,
)
