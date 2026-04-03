from dataclasses import dataclass


@dataclass(frozen=True)
class ChatModelCatalogEntry:
    """Metadata for a chat model exposed by the bot."""

    model_id: str
    display_name: str
    pricing_class: str
    capabilities: frozenset[str]
    slash_command_visible: bool = True

    @property
    def supports_penalties(self) -> bool:
        return "supports_penalties" in self.capabilities

    @property
    def supports_reasoning_effort(self) -> bool:
        return "supports_reasoning_effort" in self.capabilities

    @property
    def supports_multi_agent(self) -> bool:
        return "supports_multi_agent" in self.capabilities


MODEL_PRICING_CLASSES: dict[str, tuple[float, float, float]] = {
    "premium": (2.00, 0.20, 6.00),
    "fast": (0.20, 0.05, 0.50),
    "code_fast": (0.20, 0.02, 1.50),
    "legacy_premium": (3.00, 0.75, 15.00),
    "mini": (0.30, 0.07, 0.50),
}

CHAT_MODEL_CATALOG: tuple[ChatModelCatalogEntry, ...] = (
    ChatModelCatalogEntry(
        model_id="grok-4.20-multi-agent",
        display_name="Grok 4.20 Multi-Agent",
        pricing_class="premium",
        capabilities=frozenset({"supports_multi_agent"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4.20",
        display_name="Grok 4.20",
        pricing_class="premium",
        capabilities=frozenset(),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4.20-non-reasoning",
        display_name="Grok 4.20 Non-Reasoning",
        pricing_class="premium",
        capabilities=frozenset({"supports_penalties"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4-1-fast-reasoning",
        display_name="Grok 4.1 Fast Reasoning",
        pricing_class="fast",
        capabilities=frozenset(),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4-1-fast-non-reasoning",
        display_name="Grok 4.1 Fast Non-Reasoning",
        pricing_class="fast",
        capabilities=frozenset({"supports_penalties"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-code-fast-1",
        display_name="Grok Code Fast 1",
        pricing_class="code_fast",
        capabilities=frozenset(),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4-fast-reasoning",
        display_name="Grok 4 Fast Reasoning",
        pricing_class="fast",
        capabilities=frozenset(),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4-fast-non-reasoning",
        display_name="Grok 4 Fast Non-Reasoning",
        pricing_class="fast",
        capabilities=frozenset({"supports_penalties"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-4-0709",
        display_name="Grok 4 (0709)",
        pricing_class="legacy_premium",
        capabilities=frozenset(),
    ),
    ChatModelCatalogEntry(
        model_id="grok-3-mini",
        display_name="Grok 3 Mini",
        pricing_class="mini",
        capabilities=frozenset({"supports_reasoning_effort"}),
    ),
    ChatModelCatalogEntry(
        model_id="grok-3",
        display_name="Grok 3",
        pricing_class="legacy_premium",
        capabilities=frozenset(),
    ),
)

CHAT_MODEL_INDEX: dict[str, ChatModelCatalogEntry] = {
    entry.model_id: entry for entry in CHAT_MODEL_CATALOG
}

DEFAULT_CHAT_MODEL_ID = "grok-4.20"
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
