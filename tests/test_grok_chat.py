import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import aiohttp
import pytest

from tests.support import make_cog


class TestGrokChat:
    """Tests for the /grok chat command and follow-up orchestration."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

    async def test_chat_creates_conversation(self, cog, mock_discord_context):
        """Test that chat command creates a conversation entry."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello Grok!",
            model="grok-3",
        )

        cog._call_responses_api.assert_called_once()
        assert len(cog.conversations) == 1
        conversation = list(cog.conversations.values())[0]

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["model"] == "grok-3"
        assert payload["store"] is True
        assert cog._call_responses_api.call_args.kwargs["grok_conv_id"] == conversation.grok_conv_id
        assert conversation.grok_conv_id is not None
        assert str(UUID(conversation.grok_conv_id)) == conversation.grok_conv_id
        assert any(msg.get("role") == "user" for msg in payload["input"])

    async def test_chat_stores_response_id(self, cog, mock_discord_context):
        """Chat should store the response ID for multi-turn."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello!",
            model="grok-3",
        )

        conversation = list(cog.conversations.values())[0]
        assert conversation.previous_response_id == "resp_01XFDUDYJgAACzvnptvVoYEL"
        assert conversation.response_id_history == ["resp_01XFDUDYJgAACzvnptvVoYEL"]

    async def test_chat_with_four_tools(self, cog, mock_discord_context):
        """Chat should pass the selected four tools in the payload."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        with patch("discord_grok.cogs.grok.tooling.XAI_COLLECTION_IDS", ["collection_123"]):
            await cog.chat.callback(
                cog,
                ctx=mock_discord_context,
                prompt="Tool test",
                model="grok-4.20",
                web_search=True,
                x_search=True,
                code_execution=True,
                collections_search=True,
            )

        payload = cog._call_responses_api.call_args[0][0]
        assert "tools" in payload
        assert len(payload["tools"]) == 4
        tool_types = sorted(tool["type"] for tool in payload["tools"])
        assert tool_types == [
            "code_interpreter",
            "file_search",
            "web_search",
            "x_search",
        ]
        assert "reasoning.encrypted_content" in payload.get("include", [])

    async def test_chat_with_mcp_tool(self, cog, mock_discord_context):
        """Chat should resolve MCP presets into Responses payload tools."""
        from discord_grok.config.mcp import XaiMcpPreset

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        with patch.dict(
            "discord_grok.config.mcp.XAI_MCP_PRESETS",
            {
                "trusted": XaiMcpPreset(
                    name="trusted",
                    server_url="https://mcp.example.com/sse",
                    allowed_tools=["search", "run"],
                )
            },
            clear=True,
        ):
            await cog.chat.callback(
                cog,
                ctx=mock_discord_context,
                prompt="Use MCP",
                model="grok-4.20",
                mcp="trusted",
            )

        payload = cog._call_responses_api.call_args[0][0]
        assert [tool["type"] for tool in payload["tools"]] == ["mcp"]
        assert payload["tools"][0]["server_url"] == "https://mcp.example.com/sse"
        assert payload["tools"][0]["server_label"] == "mcp.example.com"
        assert payload["tools"][0]["allowed_tool_names"] == ["search", "run"]

        conversation = list(cog.conversations.values())[0]
        assert conversation.params.mcp_servers[0].server_url == "https://mcp.example.com/sse"
        assert conversation.params.mcp_servers[0].allowed_tool_names == ["search", "run"]

    async def test_chat_rejects_unknown_mcp_preset(self, cog, mock_discord_context):
        """Unknown MCP preset names should return a user-facing error."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Use MCP",
            mcp="unknown",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "Unknown MCP preset" in call_kwargs["embed"].description
        cog._call_responses_api.assert_not_called()

    async def test_chat_prevents_duplicate_conversations(self, cog, mock_discord_context):
        """Test that users can't start multiple conversations in the same channel."""
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

        existing_params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=mock_discord_context.author,
            channel_id=mock_discord_context.channel.id,
            conversation_id=123,
        )
        cog.conversations[123] = Conversation(params=existing_params)

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello again!",
        )

        mock_discord_context.send_followup.assert_called()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "already have an active conversation" in call_kwargs["embed"].description

    async def test_chat_default_model(self, cog, mock_discord_context):
        """Chat should use the shared default model."""
        from discord_grok.cogs.grok.command_options import DEFAULT_CHAT_MODEL_ID

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello!",
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["model"] == DEFAULT_CHAT_MODEL_ID

    async def test_chat_rejects_frequency_penalty_on_reasoning_model(
        self, cog, mock_discord_context
    ):
        """frequency_penalty should be rejected for reasoning models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3-mini",
            frequency_penalty=0.5,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "frequency_penalty" in call_kwargs["embed"].description
        assert "not supported" in call_kwargs["embed"].description

    async def test_chat_rejects_both_penalties_on_reasoning_model(self, cog, mock_discord_context):
        """Both penalties set on a reasoning model should be rejected."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3",
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "frequency_penalty" in call_kwargs["embed"].description
        assert "presence_penalty" in call_kwargs["embed"].description

    async def test_chat_allows_penalty_on_non_reasoning_model(self, cog, mock_discord_context):
        """Penalty params should be allowed for non-reasoning models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-non-reasoning",
            frequency_penalty=0.5,
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["frequency_penalty"] == 0.5

    async def test_chat_rejects_reasoning_effort_on_unsupported_model(
        self, cog, mock_discord_context
    ):
        """reasoning_effort should be rejected for models that don't support it."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3",
            reasoning_effort="high",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "reasoning_effort" in call_kwargs["embed"].description

    async def test_chat_passes_reasoning_effort_for_supported_model(
        self, cog, mock_discord_context
    ):
        """reasoning_effort should be passed to the API for grok-3-mini."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3-mini",
            reasoning_effort="high",
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["reasoning_effort"] == "high"

    async def test_chat_rejects_max_tokens_on_multi_agent(self, cog, mock_discord_context):
        """max_tokens should be rejected for multi-agent models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-multi-agent",
            max_tokens=1024,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "max_tokens" in call_kwargs["embed"].description
        assert "not supported" in call_kwargs["embed"].description

    async def test_chat_rejects_agent_count_on_non_multi_agent(self, cog, mock_discord_context):
        """agent_count should be rejected for non-multi-agent models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3",
            agent_count=4,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "agent_count" in call_kwargs["embed"].description
        assert "multi-agent" in call_kwargs["embed"].description

    async def test_chat_passes_agent_count_for_multi_agent(self, cog, mock_discord_context):
        """agent_count should be passed to the API for multi-agent models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Research quantum computing",
            model="grok-4.20-multi-agent",
            agent_count=16,
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["agent_count"] == 16
        assert "reasoning.encrypted_content" in payload["include"]

    async def test_chat_multi_agent_sets_encrypted_content(self, cog, mock_discord_context):
        """Multi-agent model should always set include with encrypted content."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-multi-agent",
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert "reasoning.encrypted_content" in payload["include"]
        assert "agent_count" not in payload

    async def test_chat_tools_set_encrypted_content(self, cog, mock_discord_context):
        """Tool-using conversations should include encrypted content."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Search for news",
            model="grok-4.20",
            web_search=True,
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert "reasoning.encrypted_content" in payload["include"]


class TestHandleNewMessageInConversation:
    """Tests for the handle_new_message_in_conversation method."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

    @pytest.fixture
    def conversation(self):
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

        starter = MagicMock()
        starter.id = 111222333
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=starter,
            channel_id=444555666,
            conversation_id=777888999,
        )
        return Conversation(
            params=params,
            previous_response_id="resp_prev",
            response_id_history=["resp_prev"],
            prompt_cache_key="test-cache-key",
            grok_conv_id="conv-cache-123",
        )

    @pytest.fixture
    def message(self, conversation):
        msg = MagicMock()
        msg.author = conversation.params.conversation_starter
        msg.author.id = 111222333
        msg.channel = MagicMock()
        msg.channel.id = 444555666
        msg.content = "Follow-up message"
        msg.attachments = []
        msg.reply = AsyncMock()
        msg.channel.typing = MagicMock()
        msg.channel.typing.return_value.__aenter__ = AsyncMock()
        msg.channel.typing.return_value.__aexit__ = AsyncMock()
        return msg

    async def test_follow_up_sends_response(self, cog, message, conversation):
        """A follow-up message should call the API and reply with embeds."""
        cog.conversations[777888999] = conversation
        cog.views[message.author] = MagicMock()

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_called_once()
        assert cog._call_responses_api.call_args.kwargs["grok_conv_id"] == "conv-cache-123"
        message.reply.assert_called()
        assert conversation.previous_response_id == "resp_01XFDUDYJgAACzvnptvVoYEL"
        assert "resp_01XFDUDYJgAACzvnptvVoYEL" in conversation.response_id_history

    async def test_paused_conversation_ignored(self, cog, message, conversation):
        """Messages in a paused conversation should be silently ignored."""
        conversation.params.paused = True

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_not_called()
        message.reply.assert_not_called()

    async def test_wrong_author_ignored(self, cog, message, conversation):
        """Messages from a non-starter user should be silently ignored."""
        message.author = MagicMock()

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_not_called()
        message.reply.assert_not_called()

    async def test_empty_content_returns_early(self, cog, message, conversation):
        """A message with no text and no attachments should return early."""
        message.content = ""
        message.attachments = []

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_not_called()

    async def test_api_error_ends_conversation(self, cog, message, conversation):
        """An API error should end the conversation and reply with an error embed."""
        cog.conversations[777888999] = conversation
        cog._call_responses_api.side_effect = aiohttp.ClientError("API failure")

        await cog.handle_new_message_in_conversation(message, conversation)

        message.reply.assert_called_once()
        embed = message.reply.call_args[1]["embed"]
        assert embed.title == "Error"
        assert 777888999 not in cog.conversations

    async def test_unsupported_image_attachment_returns_error(self, cog, message, conversation):
        """Unsupported image MIME types should return an error before calling xAI."""
        attachment = MagicMock()
        attachment.content_type = "image/webp"
        attachment.filename = "scene.webp"
        attachment.size = 1024
        attachment.url = "https://example.com/scene.webp"
        message.attachments = [attachment]

        with patch.object(cog, "_upload_file_attachment", new_callable=AsyncMock) as mock_upload:
            await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_not_called()
        mock_upload.assert_not_awaited()
        message.reply.assert_called_once()
        embed = message.reply.call_args[1]["embed"]
        assert "supports only JPEG and PNG" in embed.description

    async def test_follow_up_cancellation_propagates(self, cog, message, conversation):
        """CancelledError should not be swallowed in async follow-up handling."""
        cog._call_responses_api.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await cog.handle_new_message_in_conversation(message, conversation)

        message.reply.assert_not_called()

    async def test_follow_up_handled_api_failure_returns_user_error(
        self, cog, message, conversation
    ):
        """Handled transport failures should return a consistent user error embed."""
        cog._call_responses_api.side_effect = aiohttp.ClientError("network down")

        await cog.handle_new_message_in_conversation(message, conversation)

        message.reply.assert_called_once()
        embed = message.reply.call_args.kwargs["embed"]
        assert embed.title == "Error"
        assert "network down" in embed.description

    async def test_follow_up_unexpected_exception_is_not_silenced(self, cog, message, conversation):
        """Unexpected exceptions should propagate instead of being silently swallowed."""
        cog._call_responses_api.side_effect = RuntimeError("unexpected boom")

        with pytest.raises(RuntimeError, match="unexpected boom"):
            await cog.handle_new_message_in_conversation(message, conversation)


class TestOnMessageRouting:
    """Tests for the on_message event listener routing."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

    async def test_routes_to_correct_conversation(self, cog, mock_discord_message):
        """Message in the right channel from the right user should route to handler."""
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

        starter = mock_discord_message.author
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=starter,
            channel_id=mock_discord_message.channel.id,
            conversation_id=111,
        )
        cog.conversations[111] = Conversation(params=params, prompt_cache_key="k")
        cog.handle_new_message_in_conversation = AsyncMock()

        await cog.on_message(mock_discord_message)

        cog.handle_new_message_in_conversation.assert_awaited_once()

    async def test_wrong_channel_skipped(self, cog, mock_discord_message):
        """Message in a different channel should not route."""
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

        starter = mock_discord_message.author
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=starter,
            channel_id=999999,
            conversation_id=111,
        )
        cog.conversations[111] = Conversation(params=params, prompt_cache_key="k")
        cog.handle_new_message_in_conversation = AsyncMock()

        await cog.on_message(mock_discord_message)

        cog.handle_new_message_in_conversation.assert_not_awaited()

    async def test_wrong_author_skipped(self, cog, mock_discord_message):
        """Message from a different user should not route."""
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

        other_user = MagicMock()
        other_user.id = 999
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=other_user,
            channel_id=mock_discord_message.channel.id,
            conversation_id=111,
        )
        cog.conversations[111] = Conversation(params=params, prompt_cache_key="k")
        cog.handle_new_message_in_conversation = AsyncMock()

        await cog.on_message(mock_discord_message)

        cog.handle_new_message_in_conversation.assert_not_awaited()
