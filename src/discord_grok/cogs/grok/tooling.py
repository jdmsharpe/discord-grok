from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ...config.auth import XAI_COLLECTION_IDS
from ...config.pricing import (  # noqa: F401 — re-exported for callers
    IMAGE_PRICING,
    TOOL_INVOCATION_PRICING,
    TTS_PRICING_PER_MILLION_CHARS,
    UNKNOWN_IMAGE_MODEL_PRICING,
    VIDEO_PRICING_PER_SECOND,
)
from .command_options import (
    DEFAULT_CHAT_MODEL_ID,
    build_model_pricing_map,
    iter_slash_command_models,
)
from .models import ChatCompletionParameters, Conversation, McpServerConfig

CHUNK_TEXT_SIZE = 3500  # Maximum number of characters in each text chunk.

# Per-million-token pricing: (input_cost, cached_input_cost, output_cost).
# Built by combining the model catalog (command_options.CHAT_MODEL_CATALOG)
# with the class → price mapping (config/pricing.yaml → MODEL_PRICING_CLASSES).
MODEL_PRICING: dict[str, tuple[float, float, float]] = build_model_pricing_map()
DEFAULT_MODEL_PRICING = MODEL_PRICING[DEFAULT_CHAT_MODEL_ID]


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> float:
    """Calculate the cost in dollars for a given model and token usage.

    Cached tokens are a subset of input_tokens billed at a discounted rate.
    """
    input_price, cached_price, output_price = MODEL_PRICING.get(model, DEFAULT_MODEL_PRICING)
    non_cached = input_tokens - cached_tokens
    return (
        (non_cached / 1_000_000) * input_price
        + (cached_tokens / 1_000_000) * cached_price
        + ((output_tokens + reasoning_tokens) / 1_000_000) * output_price
    )


def calculate_tool_cost(tool_usage: dict[str, int]) -> float:
    """Calculate the cost in dollars for server-side tool invocations."""
    total = 0.0
    for key, count in tool_usage.items():
        per_1k = TOOL_INVOCATION_PRICING.get(key, 0.0)
        total += (count / 1_000) * per_1k
    return total


def calculate_tts_cost(character_count: int) -> float:
    """Calculate the cost in dollars for a TTS generation."""
    return (character_count / 1_000_000) * TTS_PRICING_PER_MILLION_CHARS


def calculate_image_cost(model: str) -> float:
    """Calculate the cost in dollars for an image generation."""
    return IMAGE_PRICING.get(model, UNKNOWN_IMAGE_MODEL_PRICING)


def calculate_video_cost(duration: int) -> float:
    """Calculate the cost in dollars for a video generation."""
    return duration * VIDEO_PRICING_PER_SECOND


# All available Grok language models
GROK_MODELS = [entry.model_id for entry in iter_slash_command_models()]

# Image generation models
GROK_IMAGE_MODELS = [
    "grok-imagine-image-pro",
    "grok-imagine-image",
]

# Video generation models
GROK_VIDEO_MODELS = [
    "grok-imagine-video",
]

# TTS voices
TTS_VOICES = ["eve", "ara", "rex", "sal", "leo"]

# Models that support frequency_penalty and presence_penalty parameters.
# Reasoning models do NOT support these parameters.
PENALTY_SUPPORTED_MODELS: set[str] = {
    entry.model_id for entry in iter_slash_command_models() if entry.supports_penalties
}

# Models that support the reasoning_effort parameter.
REASONING_EFFORT_MODELS: set[str] = {
    entry.model_id for entry in iter_slash_command_models() if entry.supports_reasoning_effort
}

# Multi-agent models that support agent_count and have special parameter constraints.
MULTI_AGENT_MODELS: set[str] = {
    entry.model_id for entry in iter_slash_command_models() if entry.supports_multi_agent
}

# Built-in tools supported by /grok chat.
TOOL_WEB_SEARCH = "web_search"
TOOL_X_SEARCH = "x_search"
TOOL_CODE_EXECUTION = "code_execution"
TOOL_COLLECTIONS_SEARCH = "collections_search"
TOOL_REMOTE_MCP = "mcp"

MAX_MCP_URL_LENGTH = 2048
MAX_MCP_ALLOWED_TOOLS = 20
MAX_MCP_LABEL_LENGTH = 50
MAX_MCP_TOOL_NAME_LENGTH = 100

# Tool names available to the Discord layer.
AVAILABLE_TOOLS = {
    TOOL_WEB_SEARCH: "Web Search",
    TOOL_X_SEARCH: "X Search",
    TOOL_CODE_EXECUTION: "Code Execution",
    TOOL_COLLECTIONS_SEARCH: "Collections Search",
    TOOL_REMOTE_MCP: "Remote MCP",
}


# Tool names managed by the conversation tool dropdown.
@dataclass(frozen=True)
class ToolRegistryEntry:
    """Single source of truth for supported tool metadata and builders."""

    canonical_name: str
    display_label: str
    responses_api_type: str
    description: str
    builder: Callable[..., dict[str, Any]]
    usage_key: str | None = None
    ui_selectable: bool = False
    supports_kwargs: bool = False
    supports_domain_filters: bool = False


# Tool registry keyed by canonical Discord-layer tool name.
TOOL_REGISTRY: dict[str, ToolRegistryEntry] = {
    TOOL_WEB_SEARCH: ToolRegistryEntry(
        canonical_name=TOOL_WEB_SEARCH,
        display_label="Web Search",
        description="Search the web in real time.",
        responses_api_type="web_search",
        builder=lambda **kw: {"type": "web_search", **kw},
        usage_key="SERVER_SIDE_TOOL_WEB_SEARCH",
        ui_selectable=True,
        supports_kwargs=True,
        supports_domain_filters=True,
    ),
    TOOL_X_SEARCH: ToolRegistryEntry(
        canonical_name=TOOL_X_SEARCH,
        display_label="X Search",
        description="Search X posts and threads.",
        responses_api_type="x_search",
        builder=lambda **kw: {"type": "x_search", **kw},
        usage_key="SERVER_SIDE_TOOL_X_SEARCH",
        ui_selectable=True,
        supports_kwargs=True,
    ),
    TOOL_CODE_EXECUTION: ToolRegistryEntry(
        canonical_name=TOOL_CODE_EXECUTION,
        display_label="Code Execution",
        description="Run Python code in a sandbox.",
        responses_api_type="code_interpreter",
        builder=lambda **kw: {"type": "code_interpreter"},  # noqa: ARG005
        usage_key="SERVER_SIDE_TOOL_CODE_EXECUTION",
        ui_selectable=True,
    ),
    TOOL_COLLECTIONS_SEARCH: ToolRegistryEntry(
        canonical_name=TOOL_COLLECTIONS_SEARCH,
        display_label="Collections Search",
        description="Search configured collections.",
        responses_api_type="file_search",
        builder=lambda **kw: {"type": "file_search", **kw},
        usage_key="SERVER_SIDE_TOOL_COLLECTIONS_SEARCH",
        ui_selectable=True,
    ),
    TOOL_REMOTE_MCP: ToolRegistryEntry(
        canonical_name=TOOL_REMOTE_MCP,
        display_label="Remote MCP",
        description="Use remote MCP server tools.",
        responses_api_type="mcp",
        builder=lambda **kw: {"type": "mcp", **kw},
        usage_key="SERVER_SIDE_TOOL_MCP",
    ),
}

SELECTABLE_TOOLS = {
    key: entry.display_label for key, entry in TOOL_REGISTRY.items() if entry.ui_selectable
}

# Tool builders that produce Responses API JSON tool dicts.
TOOL_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    key: entry.builder for key, entry in TOOL_REGISTRY.items()
}


# Display names for server_side_tool_usage keys.
TOOL_USAGE_DISPLAY_NAMES: dict[str, str] = {
    "SERVER_SIDE_TOOL_WEB_SEARCH": "Web Search",
    "SERVER_SIDE_TOOL_X_SEARCH": "X Search",
    "SERVER_SIDE_TOOL_CODE_EXECUTION": "Code Execution",
    "SERVER_SIDE_TOOL_COLLECTIONS_SEARCH": "Collections Search",
    "SERVER_SIDE_TOOL_CODE_INTERPRETER": "Code Execution",
    "SERVER_SIDE_TOOL_FILE_SEARCH": "Collections Search",
    "SERVER_SIDE_TOOL_ATTACHMENT_SEARCH": "Attachment Search",
    "SERVER_SIDE_TOOL_VIEW_X_VIDEO": "X Video",
    "SERVER_SIDE_TOOL_VIEW_IMAGE": "Image View",
    "SERVER_SIDE_TOOL_MCP": "MCP",
}


_TOOL_TYPE_TO_CANONICAL: dict[str, str] = {
    entry.responses_api_type: name for name, entry in TOOL_REGISTRY.items()
}


def resolve_tool_name(tool_config: Any) -> str | None:
    """Resolve a Responses API tool dict to its canonical tool name."""
    if not isinstance(tool_config, dict):
        return None
    tool_type = tool_config.get("type")
    if not isinstance(tool_type, str):
        return None
    canonical = _TOOL_TYPE_TO_CANONICAL.get(tool_type)
    if canonical in AVAILABLE_TOOLS:
        return canonical
    return None


def _normalize_mcp_label(hostname: str) -> str:
    label = hostname.strip().lower().rstrip(".")
    if label.startswith("www."):
        label = label[4:]
    if len(label) > MAX_MCP_LABEL_LENGTH:
        label = label[:MAX_MCP_LABEL_LENGTH]
    return label or "mcp"


def validate_mcp_server_input(
    server_url: str | None,
    allowed_tools_csv: str | None = None,
) -> tuple[McpServerConfig | None, str | None]:
    """Validate raw /grok MCP inputs and normalize them into a config object."""
    if server_url is None or not server_url.strip():
        return None, None

    normalized_url = server_url.strip()
    if len(normalized_url) > MAX_MCP_URL_LENGTH:
        return None, f"`mcp` must be {MAX_MCP_URL_LENGTH} characters or fewer."

    parsed = urlparse(normalized_url)
    if parsed.scheme != "https":
        return None, "`mcp` must be an HTTPS URL."
    if not parsed.netloc or not parsed.hostname:
        return None, "`mcp` must be a valid URL with a hostname."

    allowed_tool_names: list[str] = []
    seen_names: set[str] = set()
    if allowed_tools_csv:
        for raw_name in allowed_tools_csv.split(","):
            tool_name = raw_name.strip()
            if not tool_name:
                continue
            if len(tool_name) > MAX_MCP_TOOL_NAME_LENGTH:
                return (
                    None,
                    "`mcp_allowed_tools` entries must be "
                    f"{MAX_MCP_TOOL_NAME_LENGTH} characters or fewer.",
                )
            if tool_name in seen_names:
                continue
            seen_names.add(tool_name)
            allowed_tool_names.append(tool_name)
            if len(allowed_tool_names) > MAX_MCP_ALLOWED_TOOLS:
                return (
                    None,
                    "`mcp_allowed_tools` supports a maximum of "
                    f"{MAX_MCP_ALLOWED_TOOLS} tool names.",
                )

    return (
        McpServerConfig(
            server_url=normalized_url,
            server_label=_normalize_mcp_label(parsed.hostname),
            allowed_tool_names=allowed_tool_names,
        ),
        None,
    )


def build_mcp_tool(server: McpServerConfig) -> dict[str, Any]:
    """Build the xAI Responses API MCP tool dict for a validated server config."""
    tool = TOOL_BUILDERS[TOOL_REMOTE_MCP](
        server_url=server.server_url,
        server_label=server.server_label,
    )
    if server.allowed_tool_names:
        tool["allowed_tool_names"] = list(server.allowed_tool_names)
    return tool


def chunk_text(text: str, chunk_size: int = CHUNK_TEXT_SIZE) -> list[str]:
    """
    Splits a string into chunks of a specified size.

    Args:
        text: The string to split.
        chunk_size: The maximum size of each chunk.

    Returns:
        A list of strings, where each string is a chunk of the original text.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def truncate_text(text: str | None, max_length: int, suffix: str = "...") -> str | None:
    """
    Truncate text to max_length, adding suffix if truncated.

    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: String to append when truncated (default "...")

    Returns:
        Original text if under max_length, otherwise truncated with suffix
    """
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def format_xai_error(error: Exception) -> str:
    """Return a readable description for exceptions raised by xAI operations."""
    message = getattr(error, "message", None)
    if not isinstance(message, str) or not message.strip():
        message = str(error).strip()

    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    error_type = type(error).__name__

    details = []
    if status is not None:
        details.append(f"Status: {status}")
    if error_type and error_type != "Exception":
        details.append(f"Error: {error_type}")

    if details:
        return f"{message}\n\n" + "\n".join(details)
    return message


def resolve_selected_tools(
    selected_tool_names: list[str],
    collection_ids: list[str] | None = None,
    x_search_kwargs: dict[str, Any] | None = None,
    web_search_kwargs: dict[str, Any] | None = None,
    mcp_servers: list[McpServerConfig] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Build Responses API tool dicts for the selected tool names.

    Args:
        selected_tool_names: Canonical tool names to resolve.
        x_search_kwargs: Extra kwargs merged into the x_search tool dict.
        web_search_kwargs: Extra kwargs merged into the web_search tool dict.

    Returns:
        A tuple of (tool dicts, error message or None).
    """
    tools: list[dict[str, Any]] = []
    active_collection_ids = XAI_COLLECTION_IDS if collection_ids is None else collection_ids

    for tool_name in selected_tool_names:
        if tool_name == TOOL_REMOTE_MCP:
            continue
        registry_entry = TOOL_REGISTRY.get(tool_name)
        if registry_entry is None:
            continue

        if tool_name == TOOL_COLLECTIONS_SEARCH:
            if not active_collection_ids:
                return (
                    [],
                    "Collections search requires XAI_COLLECTION_IDS to be set in your .env.",
                )
            tools.append(registry_entry.builder(vector_store_ids=list(active_collection_ids)))
            continue

        if tool_name == TOOL_X_SEARCH and x_search_kwargs:
            tools.append(registry_entry.builder(**x_search_kwargs))
            continue

        if tool_name == TOOL_WEB_SEARCH and web_search_kwargs:
            tools.append(registry_entry.builder(**web_search_kwargs))
            continue

        tools.append(registry_entry.builder())

    for mcp_server in mcp_servers or []:
        tools.append(build_mcp_tool(mcp_server))

    return tools, None


__all__ = [
    "AVAILABLE_TOOLS",
    "CHUNK_TEXT_SIZE",
    "ChatCompletionParameters",
    "Conversation",
    "GROK_IMAGE_MODELS",
    "GROK_MODELS",
    "GROK_VIDEO_MODELS",
    "MODEL_PRICING",
    "MULTI_AGENT_MODELS",
    "PENALTY_SUPPORTED_MODELS",
    "REASONING_EFFORT_MODELS",
    "TOOL_COLLECTIONS_SEARCH",
    "TOOL_CODE_EXECUTION",
    "TOOL_REGISTRY",
    "TOOL_REMOTE_MCP",
    "TOOL_USAGE_DISPLAY_NAMES",
    "TOOL_WEB_SEARCH",
    "TOOL_X_SEARCH",
    "MAX_MCP_ALLOWED_TOOLS",
    "MAX_MCP_LABEL_LENGTH",
    "MAX_MCP_TOOL_NAME_LENGTH",
    "MAX_MCP_URL_LENGTH",
    "McpServerConfig",
    "SELECTABLE_TOOLS",
    "TTS_VOICES",
    "XAI_COLLECTION_IDS",
    "build_mcp_tool",
    "calculate_cost",
    "calculate_image_cost",
    "calculate_tool_cost",
    "calculate_tts_cost",
    "calculate_video_cost",
    "chunk_text",
    "format_xai_error",
    "validate_mcp_server_input",
    "resolve_selected_tools",
    "resolve_tool_name",
    "truncate_text",
]
