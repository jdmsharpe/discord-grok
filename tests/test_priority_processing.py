import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from discord_grok.cogs.grok.chat import handle_new_message_in_conversation
from discord_grok.cogs.grok.tooling import calculate_cost
from tests.fixtures import MOCK_RESPONSES_API_RESPONSE
from tests.support import make_cog


@pytest.mark.parametrize("served,multiplier", [("priority", 2), ("default", 1), (None, 1)])
@pytest.mark.parametrize("prompt_tokens", [1000, 200_000])
def test_priority_estimate_uses_actual_tier(served, multiplier, prompt_tokens):
    standard = calculate_cost("grok-4.7", prompt_tokens, 100, 300)
    assert calculate_cost(
        "grok-4.7", prompt_tokens, 100, 300, service_tier=served
    ) == pytest.approx(multiplier * standard)


async def test_priority_persists_into_follow_up(mock_bot, mock_discord_context):
    response = copy.deepcopy(MOCK_RESPONSES_API_RESPONSE)
    response["service_tier"] = "priority"
    cog = make_cog(mock_bot, response)
    await cog.chat.callback(
        cog, ctx=mock_discord_context, prompt="Hello", model="grok-4.7", service_tier="priority"
    )
    conversation = next(iter(cog.conversations.values()))
    assert conversation.params.service_tier == "priority"
    assert cog._call_responses_api.call_args.args[0]["service_tier"] == "priority"
    message = MagicMock()
    message.author = mock_discord_context.author
    message.channel = mock_discord_context.channel
    message.content = "Continue"
    message.attachments = []
    message.reply = AsyncMock(return_value=MagicMock(id=123, edit=AsyncMock()))
    await handle_new_message_in_conversation(cog, message, conversation)
    assert cog._call_responses_api.call_args.args[0]["service_tier"] == "priority"
