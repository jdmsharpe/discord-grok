from dataclasses import dataclass, field
from typing import Any, Callable

from discord import Member, User
from xai_sdk.tools import code_execution, web_search, x_search

CHUNK_TEXT_SIZE = 3500  # Maximum number of characters in each text chunk.

# All available Grok language models
GROK_MODELS = [
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-code-fast-1",
    "grok-4-fast-reasoning",
    "grok-4-fast-non-reasoning",
    "grok-4-0709",
    "grok-3-mini",
    "grok-3",
    "grok-2-vision-1212",
]

# Image generation models
GROK_IMAGE_MODELS = [
    "grok-imagine-image-pro",
    "grok-imagine-image",
    "grok-2-image-1212",
]

# Video generation models
GROK_VIDEO_MODELS = [
    "grok-imagine-video",
]

# Built-in tools supported by /grok converse.
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

# Tool builders that don't require runtime configuration.
TOOL_BUILDERS: dict[str, Callable[[], Any]] = {
    TOOL_WEB_SEARCH: web_search,
    TOOL_X_SEARCH: x_search,
    TOOL_CODE_EXECUTION: code_execution,
}


def resolve_tool_name(tool_config: Any) -> str | None:
    """Resolve a tool proto to its tool name."""
    which_oneof = getattr(tool_config, "WhichOneof", None)
    if not callable(which_oneof):
        return None

    tool_name = which_oneof("tool")
    if tool_name in AVAILABLE_TOOLS:
        return tool_name
    return None


@dataclass
class ChatCompletionParameters:
    """A dataclass to store the parameters for a chat completion."""

    model: str
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    tools: list[Any] = field(default_factory=list)
    conversation_starter: Member | User | None = None
    conversation_id: int | None = None
    channel_id: int | None = None
    paused: bool = False


@dataclass
class Conversation:
    """A dataclass to store conversation state."""

    params: ChatCompletionParameters
    chat: Any  # The xai_sdk Chat object


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
