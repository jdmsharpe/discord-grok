from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAppendReasoningEmbeds:
    """Tests for the append_reasoning_embeds helper."""

    def test_no_reasoning(self):
        """Empty reasoning text should not add an embed."""
        from src.xai_api import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "")
        assert len(embeds) == 0

    def test_with_reasoning(self):
        """Reasoning text should be wrapped in spoiler tags."""
        from src.xai_api import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "Some reasoning here")
        assert len(embeds) == 1
        assert embeds[0].title == "Reasoning"
        assert embeds[0].description == "||Some reasoning here||"

    def test_long_reasoning_truncated(self):
        """Long reasoning text should be truncated."""
        from src.xai_api import append_reasoning_embeds

        embeds = []
        long_text = "a" * 4000
        append_reasoning_embeds(embeds, long_text)
        assert len(embeds) == 1
        assert len(embeds[0].description) < 3600
        assert "[reasoning truncated]" in embeds[0].description


class TestAppendResponseEmbeds:
    """Tests for the append_response_embeds helper."""

    def test_short_response(self):
        """Short response should create a single embed."""
        from src.xai_api import append_response_embeds

        embeds = []
        append_response_embeds(embeds, "Hello!")
        assert len(embeds) == 1
        assert embeds[0].title == "Response"
        assert embeds[0].description == "Hello!"

    def test_long_response_chunked(self):
        """Long response should be split into multiple embeds."""
        from src.xai_api import append_response_embeds

        embeds = []
        long_text = "a" * 7500
        append_response_embeds(embeds, long_text)
        assert len(embeds) > 1
        assert embeds[0].title == "Response"
        assert "Part" in embeds[1].title

    def test_very_long_response_truncated(self):
        """Very long response should be truncated before chunking."""
        from src.xai_api import append_response_embeds

        embeds = []
        very_long_text = "a" * 25000
        append_response_embeds(embeds, very_long_text)
        total_text = "".join(e.description for e in embeds)
        assert len(total_text) < 21000


class TestXAIAPICog:
    """Tests for the xAIAPI Discord cog."""

    @pytest.fixture
    def cog(self, mock_bot):
        """Create an xAIAPI cog instance with mocked dependencies."""
        with patch("xai_sdk.AsyncClient") as mock_client_class:
            mock_client = MagicMock()

            # Mock chat
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.reasoning_content = ""
            mock_chat.sample = AsyncMock(return_value=mock_response)
            mock_chat.append = MagicMock()
            mock_chat.messages = []
            mock_client.chat = MagicMock()
            mock_client.chat.create = MagicMock(return_value=mock_chat)

            mock_client_class.return_value = mock_client

            from src.xai_api import xAIAPI

            cog = xAIAPI(bot=mock_bot)
            cog.client = mock_client
            return cog

    @pytest.mark.asyncio
    async def test_cog_initialization(self, cog, mock_bot):
        """Test that the cog initializes correctly."""
        assert cog.bot == mock_bot
        assert cog.conversations == {}
        assert cog.views == {}

    @pytest.mark.asyncio
    async def test_converse_creates_conversation(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """Test that converse command creates a conversation entry."""
        cog.client = mock_xai_client

        # Mock the channel typing context manager
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.converse.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello Grok!",
            model="grok-3",
        )

        # Verify chat was created and sampled
        mock_xai_client.chat.create.assert_called_once()

        # Verify conversation was stored
        assert len(cog.conversations) == 1

    @pytest.mark.asyncio
    async def test_converse_prevents_duplicate_conversations(
        self, cog, mock_discord_context
    ):
        """Test that users can't start multiple conversations in the same channel."""
        from src.util import ChatCompletionParameters, Conversation

        existing_params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=mock_discord_context.author,
            channel_id=mock_discord_context.channel.id,
            conversation_id=123,
        )
        cog.conversations[123] = Conversation(params=existing_params, chat=MagicMock())

        await cog.converse.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello again!",
        )

        mock_discord_context.send_followup.assert_called()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "already have an active conversation" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_on_message_ignores_bot_messages(self, cog, mock_discord_message):
        """Test that the bot ignores its own messages."""
        mock_discord_message.author = cog.bot.user

        await cog.on_message(mock_discord_message)

        mock_discord_message.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_keep_typing_can_be_cancelled(self, cog, mock_discord_context):
        """Test that the typing indicator can be cancelled."""
        import asyncio

        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        mock_discord_context.channel.typing = MagicMock(return_value=typing_cm)

        task = asyncio.create_task(cog.keep_typing(mock_discord_context.channel))

        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
