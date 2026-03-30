from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from discord import Member, User

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
    input_price, cached_price, output_price = MODEL_PRICING.get(
        model, (2.00, 0.20, 6.00)
    )
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

# Tool names available to the Discord UI.
AVAILABLE_TOOLS = {
    TOOL_WEB_SEARCH: "Web Search",
    TOOL_X_SEARCH: "X Search",
    TOOL_CODE_EXECUTION: "Code Execution",
    TOOL_COLLECTIONS_SEARCH: "Collections Search",
}

# Tool builders that produce Responses API JSON tool dicts.
TOOL_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    TOOL_WEB_SEARCH: lambda **kw: {"type": "web_search", **kw},
    TOOL_X_SEARCH: lambda **kw: {"type": "x_search", **kw},
    TOOL_CODE_EXECUTION: lambda **kw: {"type": "code_interpreter"},  # noqa: ARG005
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


@dataclass
class ChatCompletionParameters:
    """A dataclass to store the parameters for a chat completion."""

    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: str | None = None
    agent_count: int | None = None
    tools: list[Any] = field(default_factory=list)
    x_search_kwargs: dict[str, Any] = field(default_factory=dict)
    web_search_kwargs: dict[str, Any] = field(default_factory=dict)
    conversation_starter: Member | User | None = None
    conversation_id: int | None = None
    channel_id: int | None = None
    paused: bool = False


@dataclass
class Conversation:
    """A dataclass to store conversation state."""

    params: ChatCompletionParameters
    previous_response_id: str | None = None
    response_id_history: list[str] = field(default_factory=list)
    file_ids: list[str] = field(default_factory=list)
    prompt_cache_key: str = ""
    grok_conv_id: str | None = None


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
    collection_ids: list[str],
    x_search_kwargs: dict[str, Any] | None = None,
    web_search_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Build Responses API tool dicts for the selected tool names.

    Args:
        selected_tool_names: Canonical tool names to resolve.
        collection_ids: XAI_COLLECTION_IDS for file_search.
        x_search_kwargs: Extra kwargs merged into the x_search tool dict.
        web_search_kwargs: Extra kwargs merged into the web_search tool dict.

    Returns:
        A tuple of (tool dicts, error message or None).
    """
    tools: list[dict[str, Any]] = []

    for tool_name in selected_tool_names:
        if tool_name == TOOL_COLLECTIONS_SEARCH:
            if not collection_ids:
                return (
                    [],
                    "Collections search requires XAI_COLLECTION_IDS to be set in your .env.",
                )
            tools.append({
                "type": "file_search",
                "vector_store_ids": list(collection_ids),
            })
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

    return tools, None
