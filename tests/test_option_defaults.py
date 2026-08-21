"""Guard tests against slash-command descriptions advertising a stale default.

Every tunable option spells its own default into its ``description`` as a
``(default: X)`` clause, because Discord renders that description verbatim in
the slash-command picker -- it is the ONLY place a user learns what happens
when they leave the option blank. When a "promote the new default" commit
retargets the value but misses the description, nothing crashes, no test fails,
and the bot quietly tells every user the wrong thing forever. This class of
drift has shipped before and is otherwise unguarded.

These tests resolve each option's real default back to its ``OptionChoice``
display name and assert the ``(default: X)`` clause actually names it, so
retargeting a default without touching its description fails CI loudly instead
of misinforming users in production.

The clause is written for humans, so exactly four spellings are accepted:

1. The display name appears in the clause.
2. The clause equals the display name's stem -- the name with any trailing
   parenthetical removed, e.g. ``Grok Imagine Video 1.5 (Preview)`` documented
   as ``Grok Imagine Video 1.5``.
3. The clause starts with that stem -- descriptions sometimes append prose,
   e.g. ``Deep Research; Max for best reports``.
4. The raw choice value is non-empty AND appears in the clause -- descriptions
   say ``1:1`` where the display name is ``1:1 (Square)``.

Anything else names a different choice, which is drift. Two specific holes were
found by review and are now closed by anchoring EVERY one of those matches with
the ``NOT_EXTENDED`` lookahead:

* The prefix hole: a rule that accepted a display name merely *starting with*
  the clause passed a ``Foo 1`` -> ``Foo 1.5`` promotion whose description still
  said ``Foo 1``. Spellings 2 and 3 compare against the stem instead, so
  ``GPT Image 1.5`` documented as ``GPT Image 1`` is rejected.
* The superset hole, its mirror image: plain substring containment accepted a
  claim that *extends* the real name, so ``Claude Opus 5`` matched inside a
  claim of ``Claude Opus 5.1`` and promoting 5 -> 5.1 with a stale description
  still passed.

``NOT_EXTENDED`` rejects a match that continues into a longer identifier -- a
word character, a hyphen, or a dot followed by a digit -- while still allowing
ordinary sentence punctuation, because real descriptions write things like
``(default: Claude Opus 5. warning: Opus is expensive!)``. The parametrized
matcher case table below pins both holes, and the punctuation case that keeps
the anchor from being over-tightened, shut permanently.

Scope is deliberately narrow: an option is asserted over only when it has
``choices``, a non-``None`` resolved ``default``, a ``(default: X)`` clause,
AND a default that resolves to one of its own choices. An option whose
signature default is ``None`` has its effective default applied downstream in
the feature module, where introspection cannot see it; asserting over those
produces noise, and a noisy guard gets muted. An option whose default resolves
to NO choice is counted separately as unassertable rather than silently passed.
"""

import re
from unittest.mock import patch

import discord
import pytest

DEFAULT_CLAUSE_RE = re.compile(r"\(default:\s*([^)]+)\)", re.I)

# The discovered option surface, recorded exactly rather than as a floor. A
# ">= N" floor in a repo whose real count IS N is behaviourally identical to a
# non-emptiness check: it can only ever catch a total collapse, and a partial
# one (py-cord moving where options hang off subcommands, a group renamed,
# choices lost on one command) would sail through green while the guard covered
# almost nothing.
#
# NEXT CONTRIBUTOR: update these two numbers deliberately when you add or remove
# a choice-backed option that states a default. A mismatch means either a real
# change to the option surface or a discovery regression -- both deserve a human
# look, so do not "fix" a failure here by loosening the assertion.
EXPECTED_ASSERTABLE_OPTIONS = 8
EXPECTED_UNASSERTABLE_OPTIONS = 0

# Options whose stated default resolves to none of their own choices. Nothing
# is assertable about such a clause, but the hole must be a reviewed decision
# rather than an invisible gap in coverage, so any new one has to be listed
# here on purpose. Empty today; keep its length equal to
# EXPECTED_UNASSERTABLE_OPTIONS above.
KNOWN_UNASSERTABLE = ()

# Rejects a match that runs on into a longer identifier: a word character, a
# hyphen, or a dot followed by a digit. Sentence punctuation (". ", ",", ";",
# "!", ")") is deliberately still allowed through.
NOT_EXTENDED = r"(?![\w-])(?!\.\d)(?!\s+\w)"


def _import_cog_class():
    """Import the cog class with the xAI SDK client patched out."""
    with patch("xai_sdk.AsyncClient"):
        from discord_grok.cogs.grok.cog import GrokCog

        return GrokCog


def _discover_documented_defaults():
    """Discover every option that advertises a default it can be checked against.

    Walks the cog class for ``SlashCommandGroup`` attributes, then their
    subcommands, then each subcommand's resolved options -- so an option added
    to any future command is guarded automatically with no edits here.

    Returns ``(params, unassertable)``: pytest params of
    ``(label, claimed, default, display_name)`` for options that can be
    checked, and the labels of options that state a default which matches none
    of their own choices. That second group is a separate defect this guard
    does not adjudicate, but it is reported rather than dropped on the floor.
    """
    cog_cls = _import_cog_class()
    params = []
    unassertable = []
    for group in vars(cog_cls).values():
        if not isinstance(group, discord.SlashCommandGroup):
            continue
        for subcommand in group.subcommands:
            for opt in getattr(subcommand, "options", None) or []:
                choices = getattr(opt, "choices", None) or []
                default = getattr(opt, "default", None)
                if not choices or default is None:
                    continue
                match = DEFAULT_CLAUSE_RE.search(getattr(opt, "description", "") or "")
                if not match:
                    continue
                label = f"{subcommand.qualified_name}:{opt.name}"
                display_name = next((c.name for c in choices if c.value == default), None)
                if display_name is None:
                    unassertable.append(label)
                    continue
                params.append(
                    pytest.param(label, match.group(1).strip(), default, display_name, id=label)
                )
    return params, unassertable


DOCUMENTED_DEFAULTS, UNASSERTABLE_OPTIONS = _discover_documented_defaults()


def _description_names_default(display_name, raw_value, claimed):
    """Whether a ``(default: X)`` clause legitimately names the real default.

    Accepts the four spellings documented in the module docstring and nothing
    else. Every one of them is anchored with ``NOT_EXTENDED`` so a claim that
    extends the real identifier (``Claude Opus 5`` matched inside a claim of
    ``Claude Opus 5.1``) is rejected, while a claim followed by ordinary
    sentence punctuation still passes. Spellings 2 and 3 compare against the
    *stem* -- the name with any trailing parenthetical removed -- rather than
    letting the display name merely start with the clause, which is what keeps
    ``GPT Image 1.5`` documented as ``GPT Image 1`` rejected. The raw value
    check is guarded on a non-empty value, since an empty value would otherwise
    match everywhere and accept arbitrary wrong text.
    """
    name = (display_name or "").strip().lower()
    value = str(raw_value or "").strip().lower()
    claim = (claimed or "").strip().lower()
    if not name or not claim:
        return False
    stem = re.sub(r"\s*\(.*", "", name).strip()
    if re.search(re.escape(name) + NOT_EXTENDED, claim):
        return True
    if stem and claim == stem:
        return True
    if stem and re.match(re.escape(stem) + NOT_EXTENDED, claim):
        return True
    return bool(value and re.search(re.escape(value) + NOT_EXTENDED, claim))


# Fixed cases for the matcher itself. These run no matter what discovery finds,
# so the rule can never be rendered vacuous by a refactor -- and they pin the
# exact boundary between "a human shortened the name" and "the description
# names a different model".
MATCHER_CASES = [
    (
        "Gemini 3.7 Flash",
        "gemini-3.7-flash",
        "Gemini 3.7 Flash Pro",
        False,
    ),
    pytest.param("GPT Image 2", "gpt-image-2", "GPT Image 1.5", False, id="real-drift"),
    pytest.param(
        "GPT Image 1.5", "gpt-image-1.5", "GPT Image 1", False, id="prefix-superset-drift"
    ),
    pytest.param("Claude Opus 5", "claude-opus-5", "Claude Opus 5.1", False, id="superset-drift"),
    pytest.param(
        "Claude Opus 5",
        "claude-opus-5",
        "Claude Opus 5. warning: Opus is expensive!",
        True,
        id="sentence-punctuation-after-name",
    ),
    pytest.param(
        "Grok Imagine Video 1.5 (Preview)",
        "grok-imagine-video-1.5-preview",
        "Grok Imagine Video 1.5",
        True,
        id="trailing-parenthetical-trimmed",
    ),
    pytest.param(
        "Deep Research (Apr 2026)",
        "deep-research-preview-04-2026",
        "Deep Research; Max for best reports",
        True,
        id="prose-after-the-stem",
    ),
    pytest.param("Square (1:1)", "1:1", "1:1", True, id="description-uses-raw-value"),
    pytest.param("Kore (Firm)", "Kore", "Kore", True, id="value-spelling"),
    pytest.param(
        "Gemini 3.7 Flash", "gemini-3.7-flash", "Gemini 3.6 Flash", False, id="real-drift-older"
    ),
    pytest.param("Anything", "", "total nonsense", False, id="empty-value-must-not-accept"),
    pytest.param(
        "Gemini 3.1 Flash Preview TTS",
        "gemini-3.1-flash-tts-preview",
        "Gemini 2.5 Flash Preview TTS",
        False,
        id="real-drift-tts",
    ),
]


@pytest.mark.parametrize(("display_name", "value", "claimed", "expected"), MATCHER_CASES)
def test_matcher_accepts_only_legitimate_spellings(display_name, value, claimed, expected):
    """The acceptance rule itself, exercised directly on known-good/known-bad cases.

    Two cases matter most, and they are mirror images. ``prefix-superset-drift``
    is a ``Foo 1.5`` default left documented as ``Foo 1``; ``superset-drift`` is
    a ``Foo 1`` default promoted to ``Foo 1.5`` with the old clause still in
    place. Both used to pass and must never pass again.
    ``sentence-punctuation-after-name`` is the counterweight: the anchor that
    closes those holes must not reject a real description that continues into a
    sentence. ``empty-value-must-not-accept`` pins the third hole, where an
    empty choice value made every claim vacuously acceptable.
    """
    assert _description_names_default(display_name, value, claimed) is expected, (
        f"matcher regression: claimed={claimed!r} vs display_name={display_name!r} "
        f"(value={value!r}) should be {'accepted' if expected else 'rejected'}"
    )


def test_discovery_finds_the_expected_options():
    """Discovery must reach the whole option surface -- exactly, not merely at all.

    An exact count is the point: a partial collapse leaves the parametrized
    guard passing green while covering almost nothing, and only a count that
    must EQUAL the recorded one makes that impossible to miss.
    """
    assert (len(DOCUMENTED_DEFAULTS), len(UNASSERTABLE_OPTIONS)) == (
        EXPECTED_ASSERTABLE_OPTIONS,
        EXPECTED_UNASSERTABLE_OPTIONS,
    ), (
        f"discovery found {len(DOCUMENTED_DEFAULTS)} assertable and "
        f"{len(UNASSERTABLE_OPTIONS)} unassertable choice-backed option(s) with a "
        f"'(default: X)' clause on GrokCog, but {EXPECTED_ASSERTABLE_OPTIONS} and "
        f"{EXPECTED_UNASSERTABLE_OPTIONS} are recorded. Either the option surface really "
        "changed -- update the numbers deliberately -- or discovery regressed and this "
        "guard is now covering less than it claims. Do not loosen the assertion to make "
        f"it pass. Assertable: {[p.values[0] for p in DOCUMENTED_DEFAULTS]}. "
        f"Unassertable (default matches no choice): {sorted(UNASSERTABLE_OPTIONS) or 'none'}."
    )


def test_unassertable_options_are_acknowledged():
    """No option silently escapes the guard by defaulting to a value it does not offer.

    An option with choices whose resolved default matches none of them cannot
    have its ``(default: X)`` clause checked against anything -- but that is a
    defect in its own right, not a free pass. Any such option must be listed in
    ``KNOWN_UNASSERTABLE`` deliberately, so the gap shows up in review.
    """
    assert sorted(UNASSERTABLE_OPTIONS) == sorted(KNOWN_UNASSERTABLE), (
        f"unassertable options changed: {sorted(UNASSERTABLE_OPTIONS)} vs the reviewed "
        f"{sorted(KNOWN_UNASSERTABLE)}. An option whose default matches none of its own "
        "choices cannot be checked; fix the default, or add it here on purpose."
    )


@pytest.mark.parametrize(("label", "claimed", "default", "display_name"), DOCUMENTED_DEFAULTS)
def test_description_default_matches_option_default(label, claimed, default, display_name):
    """Each ``(default: X)`` clause names the choice the option actually defaults to."""
    assert _description_names_default(display_name, default, claimed), (
        f"option {label} advertises '(default: {claimed})' but actually defaults "
        f"to {default!r} ({display_name!r}). Discord shows that description "
        "verbatim in the slash-command picker, so every user is told the wrong "
        "default. Update the description to name the real default."
    )
