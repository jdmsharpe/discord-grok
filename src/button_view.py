"""Legacy button_view shim; re-exports the namespaced view."""

import warnings

from discord_grok.cogs.grok.views import ButtonView

warnings.warn(
    "Importing ButtonView from button_view is deprecated; use discord_grok.cogs.grok.views instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ButtonView"]
