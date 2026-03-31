from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from ...config.auth import XAI_COLLECTION_IDS
from .models import ChatCompletionParameters, Conversation, McpServerConfig

CHUNK_TEXT_SIZE = 3500  # Maximum number of characters in each text chunk.

# Per-million-token pricing: (input_cost, cached_input_cost, output_cost)
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "grok-4.20-multi-agent": (2.00, 0.20, 6.00),
    "grok-4.20": (2.00, 0.20, 6.00),
    "grok-4.20-non-reasoning": (2.00, 0.20, 6.00),
    "grok-4-1-fast-reasoning": (0.20, 0.05, 0.50),
    "grok-4-1-fast-non-reasoning": (0.20, 0.05, 0.50),
    "grok-code-fast-1": (0.20, 0.02, 1.50),
    "grok-4-fast-reasoning": (0.20, 0.05, 0.50),
    "grok-4-fast-non-reasoning": (0.20, 0.05, 0.50),
    "grok-4-0709": (3.00, 0.75, 15.00),
    "grok-3-mini": (0.30, 0.07, 0.50),
    "grok-3": (3.00, 0.75, 15.00),
}

# Flat per-image pricing
IMAGE_PRICING: dict[str, float] = {
    "grok-imagine-image-pro": 0.07,
    "grok-imagine-image": 0.02,
}

# Per-second video pricing
VIDEO_PRICING_PER_SECOND: float = 0.05

# Per-1k-invocations pricing for server-side tools
TOOL_INVOCATION_PRICING: dict[str, float] = {
    "SERVER_SIDE_TOOL_WEB_SEARCH": 5.00,
    "SERVER_SIDE_TOOL_X_SEARCH": 5.00,
    "SERVER_SIDE_TOOL_CODE_EXECUTION": 5.00,
    "SERVER_SIDE_TOOL_CODE_INTERPRETER": 5.00,
    "SERVER_SIDE_TOOL_COLLECTIONS_SEARCH": 2.50,
    "SERVER_SIDE_TOOL_FILE_SEARCH": 2.50,
    "SERVER_SIDE_TOOL_ATTACHMENT_SEARCH": 10.00,
}

# TTS pricing: dollars per million characters
TTS_PRICING_PER_MILLION_CHARS: float = 4.20


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
    input_price, cached_price, output_price = MODEL_PRICING.get(model, (2.00, 0.20, 6.00))
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
    return IMAGE_PRICING.get(model, 0.07)


def calculate_video_cost(duration: int) -> float:
    """Calculate the cost in dollars for a video generation."""
    return duration * VIDEO_PRICING_PER_SECOND


# All available Grok language models
GROK_MODELS = [
    "grok-4.20-multi-agent",
    "grok-4.20",
    "grok-4.20-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-code-fast-1",
    "grok-4-fast-reasoning",
    "grok-4-fast-non-reasoning",
    "grok-4-0709",
    "grok-3-mini",
    "grok-3",
]

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
    "grok-4.20-non-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-4-fast-non-reasoning",
}

# Models that support the reasoning_effort parameter.
REASONING_EFFORT_MODELS: set[str] = {"grok-3-mini"}

# Multi-agent models that support agent_count and have special parameter constraints.
MULTI_AGENT_MODELS: set[str] = {"grok-4.20-multi-agent"}

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
SELECTABLE_TOOLS = {
    TOOL_WEB_SEARCH: AVAILABLE_TOOLS[TOOL_WEB_SEARCH],
    TOOL_X_SEARCH: AVAILABLE_TOOLS[TOOL_X_SEARCH],
    TOOL_CODE_EXECUTION: AVAILABLE_TOOLS[TOOL_CODE_EXECUTION],
    TOOL_COLLECTIONS_SEARCH: AVAILABLE_TOOLS[TOOL_COLLECTIONS_SEARCH],
}

# Tool builders that produce Responses API JSON tool dicts.
TOOL_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    TOOL_WEB_SEARCH: lambda **kw: {"type": "web_search", **kw},
    TOOL_X_SEARCH: lambda **kw: {"type": "x_search", **kw},
    TOOL_CODE_EXECUTION: lambda **kw: {"type": "code_interpreter"},  # noqa: ARG005
    TOOL_REMOTE_MCP: lambda **kw: {"type": "mcp", **kw},
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
    "web_search": TOOL_WEB_SEARCH,
    "x_search": TOOL_X_SEARCH,
    "code_interpreter": TOOL_CODE_EXECUTION,
    "file_search": TOOL_COLLECTIONS_SEARCH,
    "mcp": TOOL_REMOTE_MCP,
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
        if tool_name == TOOL_COLLECTIONS_SEARCH:
            if not active_collection_ids:
                return (
                    [],
                    "Collections search requires XAI_COLLECTION_IDS to be set in your .env.",
                )
            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": list(active_collection_ids),
                }
            )
            continue

        if tool_name == TOOL_X_SEARCH and x_search_kwargs:
            tools.append({"type": "x_search", **x_search_kwargs})
            continue

        if tool_name == TOOL_WEB_SEARCH and web_search_kwargs:
            tools.append({"type": "web_search", **web_search_kwargs})
            continue

        tool_builder = TOOL_BUILDERS.get(tool_name)
        if tool_builder is None:
            continue
        tools.append(tool_builder())

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
