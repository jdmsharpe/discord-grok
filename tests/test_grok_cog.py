import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.support import make_cog


class TestGrokCog:
    """Tests for the GrokCog Discord cog."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

    async def test_cog_initialization(self, cog, mock_bot):
        """Test that the cog initializes correctly."""
        assert cog.bot == mock_bot
        assert cog.conversations == {}
        assert cog.views == {}

    async def test_on_message_ignores_bot_messages(self, cog, mock_discord_message):
        """Test that the bot ignores its own messages."""
        mock_discord_message.author = cog.bot.user

        await cog.on_message(mock_discord_message)

        mock_discord_message.reply.assert_not_called()

    async def test_keep_typing_can_be_cancelled(self, cog, mock_discord_context):
        """Test that the typing indicator can be cancelled."""
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        mock_discord_context.channel.typing = MagicMock(return_value=typing_cm)

        task = asyncio.create_task(cog.keep_typing(mock_discord_context.channel))

        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
