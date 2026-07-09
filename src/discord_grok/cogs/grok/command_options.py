from dataclasses import dataclass

from ...config.pricing import MODEL_PRICING_CLASSES


@dataclass(frozen=True)
class ChatModelCatalogEntry:
    """Metadata for a chat model exposed by the bot."""

    model_id: str
    display_name: str
    pricing_class: str
    capabilities: frozenset[str]
    reasoning_efforts: frozenset[str] = frozenset()
    slash_command_visible: bool = True

    @property
    def supports_penalties(self) -> bool:
        return "supports_penalties" in self.capabilities

    @property
    def supports_reasoning_effort(self) -> bool:
        return bool(self.reasoning_efforts)

    @property
    def supports_multi_agent(self) -> bool:
        return "supports_multi_agent" in self.capabilities


CHAT_MODEL_CATALOG: tuple[ChatModelCatalogEntry, ...] = (
    ChatModelCatalogEntry(
        model_id="grok-4.5",
        display_name="Grok 4.5",
        pricing_class="grok_4_5",
        # Reasoning models reject presence/frequency penalties and `stop`, and grok-4.5's
        # reasoning cannot be disabled — omitting "none" here is deliberate, not an oversight.
        capabilities=frozenset(),
        reasoning_efforts=frozenset({"low", "medium", "high"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4.3",
        display_name="Grok 4.3",
        pricing_class="flagship",
        capabilities=frozenset(),
        reasoning_efforts=frozenset({"none", "low", "medium", "high"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4.20-multi-agent",
        display_name="Grok 4.20 Multi-Agent",
        pricing_class="flagship",
        capabilities=frozenset({"supports_multi_agent"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4.20",
        display_name="Grok 4.20",
        pricing_class="flagship",
        capabilities=frozenset(),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4.20-non-reasoning",
        display_name="Grok 4.20 Non-Reasoning",
        pricing_class="flagship",
        capabilities=frozenset({"supports_penalties"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-build-0.1",
        display_name="Grok Build 0.1",
        pricing_class="build",
        capabilities=frozenset(),
    ),
)

CHAT_MODEL_INDEX: dict[str, ChatModelCatalogEntry] = {
    entry.model_id: entry for entry in CHAT_MODEL_CATALOG
}

DEFAULT_CHAT_MODEL_ID = "grok-4.3"
DEFAULT_CHAT_MODEL_ENTRY = CHAT_MODEL_INDEX[DEFAULT_CHAT_MODEL_ID]


def build_model_pricing_map() -> dict[str, tuple[float, float, float]]:
    """Build concrete per-model token pricing from pricing classes."""
    return {
        entry.model_id: MODEL_PRICING_CLASSES[entry.pricing_class] for entry in CHAT_MODEL_CATALOG
    }


def iter_slash_command_models() -> tuple[ChatModelCatalogEntry, ...]:
    """Return models that should be exposed as slash-command choices."""
    return tuple(entry for entry in CHAT_MODEL_CATALOG if entry.slash_command_visible)


def generate_model_markdown_lines() -> list[str]:
    """Render a compact markdown list for README generation workflows."""
    return [
        f"- `{entry.model_id}` — {entry.display_name} ({entry.pricing_class})"
        for entry in iter_slash_command_models()
    ]


__all__ = [
    "CHAT_MODEL_CATALOG",
    "CHAT_MODEL_INDEX",
    "DEFAULT_CHAT_MODEL_ENTRY",
    "DEFAULT_CHAT_MODEL_ID",
    "MODEL_PRICING_CLASSES",
    "ChatModelCatalogEntry",
    "build_model_pricing_map",
    "generate_model_markdown_lines",
    "iter_slash_command_models",
]
