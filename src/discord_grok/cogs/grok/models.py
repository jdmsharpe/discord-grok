from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from discord import Member, User


class CitationInfo(TypedDict):
    url: str
    source: Literal["web", "x", "collections"]


class ToolInfo(TypedDict):
    citations: list[CitationInfo]


@dataclass
class McpServerConfig:
    """Validated MCP server configuration persisted with a conversation."""

    server_url: str
    server_label: str
    allowed_tool_names: list[str] = field(default_factory=list)


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
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
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


__all__ = [
    "ChatCompletionParameters",
    "CitationInfo",
    "Conversation",
    "McpServerConfig",
    "ToolInfo",
]
