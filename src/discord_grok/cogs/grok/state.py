from __future__ import annotations

import contextlib
from datetime import date

from discord import Member, User

from .client import cleanup_conversation_files
from .tooling import SELECTABLE_TOOLS, resolve_selected_tools, resolve_tool_name
from .views import ButtonView


def track_daily_cost(cog, user_id: int, cost: float) -> float:
    """Add a cost to the user's daily total and return the new daily total."""
    key = (user_id, date.today().isoformat())
    cog.daily_costs[key] = cog.daily_costs.get(key, 0.0) + cost
    return cog.daily_costs[key]


def log_chat_cost(
    cog,
    user_id: int,
    model: str,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    image_tokens: int,
    tool_usage: dict[str, int],
    request_cost: float,
    daily_cost: float,
) -> None:
    """Log a structured cost line for a chat API call."""
    cog.logger.info(
        "COST | command=chat | user=%s | model=%s | input=%d | cached=%d | output=%d"
        " | reasoning=%d | image=%d | tool_usage=%s | cost=$%.4f | daily=$%.4f",
        user_id,
        model,
        input_tokens,
        cached_tokens,
        output_tokens,
        reasoning_tokens,
        image_tokens,
        tool_usage or {},
        request_cost,
        daily_cost,
    )


async def strip_previous_view(cog, user: Member | User) -> None:
    """Edit the last message that had buttons to remove its view."""
    prev = cog.last_view_messages.pop(user, None)
    if prev is not None:
        with contextlib.suppress(Exception):
            await prev.edit(view=None)


async def end_conversation(cog, conversation_id: int) -> None:
    """End a conversation and clean up associated resources."""
    conversation = cog.conversations.pop(conversation_id, None)
    if conversation is None:
        return
    starter = conversation.params.conversation_starter
    if starter is not None:
        await strip_previous_view(cog, starter)
        cog.views.pop(starter, None)
    await cleanup_conversation_files(cog, conversation)


def resolve_tools_for_view(
    cog,
    selected_values: list[str],
    conversation,
) -> tuple[set[str], str | None]:
    """Resolve tool selection from ButtonView."""
    tools, error_message = resolve_selected_tools(
        selected_values,
        x_search_kwargs=conversation.params.x_search_kwargs,
        web_search_kwargs=conversation.params.web_search_kwargs,
        mcp_servers=conversation.params.mcp_servers,
    )
    if error_message:
        return set(), error_message

    conversation.params.tools = tools
    active_names = {
        tool_name
        for tool in tools
        if (tool_name := resolve_tool_name(tool)) in SELECTABLE_TOOLS
    }
    return active_names, None


def create_button_view(
    cog,
    *,
    user,
    conversation_id: int,
    initial_tools: list[dict[str, object]] | None,
) -> ButtonView:
    """Create the provider-local button view for an active conversation."""
    view = ButtonView(
        conversation_starter=user,
        conversation_id=conversation_id,
        initial_tools=initial_tools,
        get_conversation=lambda cid: cog.conversations.get(cid),
        on_regenerate=cog.handle_new_message_in_conversation,
        on_stop=cog.end_conversation,
        on_tools_changed=lambda values, conversation: resolve_tools_for_view(
            cog,
            values,
            conversation,
        ),
    )
    cog.views[user] = view
    return view


__all__ = [
    "create_button_view",
    "end_conversation",
    "log_chat_cost",
    "resolve_tools_for_view",
    "strip_previous_view",
    "track_daily_cost",
]
