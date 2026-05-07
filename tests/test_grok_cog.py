import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.support import make_cog


def _serialize_command_group_payload(group):
    return {
        "name": group.name,
        "description": group.description,
        "options": [
            {
                "name": command.name,
                "description": command.description,
                "options": [
                    option.to_dict()
                    for option in command.options
                    if option.input_type is not None
                ],
                "type": 1,
                "nsfw": False,
            }
            for command in group.subcommands
        ],
        "nsfw": False,
    }


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

    def test_registered_command_groups_fit_discord_size_limit(self, cog):
        """Discord rejects any single top-level command payload over 8000 bytes."""

        commands_by_name = {command.name: command for command in cog.get_commands()}

        assert set(commands_by_name) == {"grok", "grok-media", "grok-tools"}
        assert [command.name for command in commands_by_name["grok"].subcommands] == [
            "check_permissions",
            "chat",
        ]
        assert [command.name for command in commands_by_name["grok-media"].subcommands] == [
            "image",
            "video",
        ]
        assert [command.name for command in commands_by_name["grok-tools"].subcommands] == [
            "tts",
        ]

        payload_sizes = {
            name: len(
                json.dumps(
                    _serialize_command_group_payload(command),
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for name, command in commands_by_name.items()
        }

        assert payload_sizes["grok"] < 8000
        assert payload_sizes["grok-media"] < 8000
        assert payload_sizes["grok-tools"] < 8000

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
