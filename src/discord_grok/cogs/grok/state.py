from __future__ import annotations

import contextlib
from datetime import date, datetime, timedelta, timezone

from discord import Member, User

from .client import cleanup_conversation_files
from .tooling import SELECTABLE_TOOLS, resolve_selected_tools, resolve_tool_name
from .views import ButtonView

MAX_ACTIVE_CONVERSATIONS = 100
CONVERSATION_TTL = timedelta(hours=12)
DAILY_COST_RETENTION_DAYS = 30


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _extract_daily_total(value: float | tuple[float, datetime]) -> float:
    return value[0] if isinstance(value, tuple) else value


def track_daily_cost(cog, user_id: int, cost: float) -> float:
    """Add a cost to the user's daily total and return the new daily total."""
    prune_daily_costs(cog)
    key = (user_id, date.today().isoformat())
    current_total = _extract_daily_total(cog.daily_costs.get(key, 0.0))
    new_total = current_total + cost
    cog.daily_costs[key] = (new_total, _now_utc())
    return new_total


def prune_daily_costs(cog) -> None:
    cutoff = date.today() - timedelta(days=DAILY_COST_RETENTION_DAYS)
    expired_keys = [key for key in cog.daily_costs if date.fromisoformat(key[1]) < cutoff]
    for key in expired_keys:
        cog.daily_costs.pop(key, None)


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
    await prune_runtime_state(cog)


async def prune_runtime_state(cog) -> None:
    """Evict stale conversations, cascade-clean views, and prune old daily costs."""
    now = _now_utc()

    stale_conversation_ids = [
        cid
        for cid, conversation in cog.conversations.items()
        if now - conversation.updated_at > CONVERSATION_TTL
    ]

    active_conversations = [
        (cid, conversation)
        for cid, conversation in cog.conversations.items()
        if cid not in stale_conversation_ids
    ]
    overflow = len(active_conversations) - MAX_ACTIVE_CONVERSATIONS
    if overflow > 0:
        active_conversations.sort(key=lambda item: item[1].updated_at)
        stale_conversation_ids.extend(cid for cid, _ in active_conversations[:overflow])

    for cid in dict.fromkeys(stale_conversation_ids):
        conversation = cog.conversations.pop(cid, None)
        if conversation is None:
            continue
        starter = conversation.params.conversation_starter
        if starter is not None:
            await strip_previous_view(cog, starter)
            cog.views.pop(starter, None)
        with contextlib.suppress(Exception):
            await cleanup_conversation_files(cog, conversation)

    orphaned_users = [
        user for user, view in cog.views.items() if view.conversation_id not in cog.conversations
    ]
    for user in orphaned_users:
        await strip_previous_view(cog, user)
        cog.views.pop(user, None)

    prune_daily_costs(cog)


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
        tool_name for tool in tools if (tool_name := resolve_tool_name(tool)) in SELECTABLE_TOOLS
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
    "CONVERSATION_TTL",
    "DAILY_COST_RETENTION_DAYS",
    "MAX_ACTIVE_CONVERSATIONS",
    "create_button_view",
    "end_conversation",
    "log_chat_cost",
    "prune_daily_costs",
    "prune_runtime_state",
    "resolve_tools_for_view",
    "strip_previous_view",
    "track_daily_cost",
]
