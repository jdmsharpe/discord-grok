"""Guard tests for Discord's 25-static-choices-per-option cap.

Discord rejects any slash-command option carrying more than 25 static
``choices`` with API error 50035. py-cord syncs every registered command in a
single all-or-nothing bulk ``PUT`` on ``on_connect``, so ONE over-limit list
silently aborts slash-command registration for EVERY cog in the bot -- not just
the offending command. That makes this the most catastrophic silent failure in
the repo, and it is otherwise unguarded.

These tests assert ``len(choices) <= 25`` across two surfaces so that adding a
model to the catalog (which feeds the generated ``CHAT_MODEL_CHOICES``
comprehension) or padding any inline ``choices=[...]`` list fails CI loudly
instead of silently breaking command registration in production:

1. Every module-level ``*_CHOICES`` list in ``cog.py`` -- counting the already
   RESOLVED module-level object handles the generated comprehension over
   ``iter_slash_command_models()`` correctly with zero AST machinery.
2. Every resolved per-option ``choices`` on the actual constructed cog -- this
   additionally catches choices defined inline on ``@option(...)`` decorators
   (e.g. image ``aspect_ratio``) that are not bound to a module-level name.
"""

from unittest.mock import MagicMock, patch

import pytest

DISCORD_MAX_STATIC_CHOICES = 25


def _import_cog_module():
    """Import the cog module with the xAI SDK client patched out."""
    with patch("xai_sdk.AsyncClient"):
        from discord_grok.cogs.grok import cog as cog_module

        return cog_module


def _module_choice_lists():
    """Discover every module-level list named ``*_CHOICES`` in the cog module.

    Returns ``(id, name, list)`` triples for pytest parametrization. Counting
    the resolved module-level object means the generated
    ``CHAT_MODEL_CHOICES`` comprehension is measured at its true runtime length
    with no AST parsing.
    """
    cog_module = _import_cog_module()
    triples = []
    for name in sorted(vars(cog_module)):
        if not name.endswith("_CHOICES"):
            continue
        value = getattr(cog_module, name)
        if isinstance(value, list):
            triples.append(pytest.param(name, value, id=name))
    return triples


def _option_choice_lists():
    """Discover every resolved per-option ``choices`` on the constructed cog.

    Building the real cog and walking its commands captures choices defined
    inline on ``@option(...)`` decorators in addition to the module-level
    ``*_CHOICES`` constants, so newly added inline menus are guarded too.
    """
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 1
    with patch("xai_sdk.AsyncClient"):
        from discord_grok import GrokCog

        cog = GrokCog(bot=bot)

    params = []
    for command in cog.walk_commands():
        for opt in getattr(command, "options", None) or []:
            choices = getattr(opt, "choices", None)
            if choices:
                label = f"{command.qualified_name}:{opt.name}"
                params.append(pytest.param(label, list(choices), id=label))
    return params


MODULE_CHOICE_LISTS = _module_choice_lists()
OPTION_CHOICE_LISTS = _option_choice_lists()


def test_choice_lists_were_discovered():
    """Sanity check: the discovery helpers must find something to guard.

    If a refactor moves or renames every choices list, an empty parametrize set
    would make the cap assertions vacuously pass. Fail loudly instead.
    """
    assert MODULE_CHOICE_LISTS, "no module-level *_CHOICES lists discovered in cog.py"
    assert OPTION_CHOICE_LISTS, "no per-option choices discovered on the cog"


@pytest.mark.parametrize(("name", "choices"), MODULE_CHOICE_LISTS)
def test_module_choices_under_discord_cap(name, choices):
    """Each module-level ``*_CHOICES`` list stays within Discord's 25 cap."""
    count = len(choices)
    assert count <= DISCORD_MAX_STATIC_CHOICES, (
        f"{name} has {count} choices (> {DISCORD_MAX_STATIC_CHOICES}). "
        "Discord rejects the bulk command sync with API error 50035, which "
        "SILENTLY aborts slash-command registration for EVERY cog in the bot."
    )


@pytest.mark.parametrize(("label", "choices"), OPTION_CHOICE_LISTS)
def test_option_choices_under_discord_cap(label, choices):
    """Each resolved slash-command option (incl. inline) stays within the cap."""
    count = len(choices)
    assert count <= DISCORD_MAX_STATIC_CHOICES, (
        f"option {label} has {count} choices (> {DISCORD_MAX_STATIC_CHOICES}). "
        "Discord rejects the bulk command sync with API error 50035, which "
        "SILENTLY aborts slash-command registration for EVERY cog in the bot."
    )
