from unittest.mock import AsyncMock, MagicMock

import pytest
from discord.ui import Select

from src.util import TOOL_BUILDERS


@pytest.mark.asyncio
class TestButtonView:
    @pytest.fixture
    def conversation_starter(self):
        user = MagicMock()
        user.id = 12345
        return user

    @pytest.fixture
    def cog(self, conversation_starter):
        cog = MagicMock()
        cog.conversations = {}
        cog.resolve_selected_tools = MagicMock(return_value=([], None))
        cog._apply_tools_to_chat = MagicMock()

        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.model = "grok-4-1-fast-reasoning"
        conversation.params.tools = []
        conversation.chat = MagicMock()
        cog.conversations[111] = conversation
        return cog

    async def test_init_adds_tool_select(self, cog, conversation_starter):
        from src.button_view import ButtonView

        view = ButtonView(
            cog=cog,
            conversation_starter=conversation_starter,
            conversation_id=111,
        )

        selects = [item for item in view.children if isinstance(item, Select)]
        assert len(selects) == 1
        assert selects[0].min_values == 0
        assert selects[0].max_values == 4

    async def test_init_with_initial_tools_sets_defaults(
        self, cog, conversation_starter
    ):
        from src.button_view import ButtonView

        view = ButtonView(
            cog=cog,
            conversation_starter=conversation_starter,
            conversation_id=111,
            initial_tools=[
                TOOL_BUILDERS["web_search"](),
                TOOL_BUILDERS["code_execution"](),
            ],
        )

        tool_select = next(item for item in view.children if isinstance(item, Select))
        defaults = {option.value: option.default for option in tool_select.options}
        assert defaults["web_search"] is True
        assert defaults["code_execution"] is True
        assert defaults["x_search"] is False
        assert defaults["collections_search"] is False

    @pytest.mark.asyncio
    async def test_tool_select_callback_updates_tools(self, cog, conversation_starter):
        from src.button_view import ButtonView

        view = ButtonView(
            cog=cog,
            conversation_starter=conversation_starter,
            conversation_id=111,
        )

        selected_tools = [
            TOOL_BUILDERS["web_search"](),
            TOOL_BUILDERS["code_execution"](),
        ]
        cog.resolve_selected_tools.return_value = (selected_tools, None)

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.data = {"values": ["web_search", "code_execution"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction)

        conversation = cog.conversations[111]
        assert conversation.params.tools == selected_tools
        cog._apply_tools_to_chat.assert_called_once_with(
            conversation.chat, selected_tools
        )

        call_args = interaction.response.send_message.call_args
        assert "Tools updated" in call_args.args[0]
        assert call_args.kwargs["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_tool_select_callback_rejects_non_owner(self, cog, conversation_starter):
        from src.button_view import ButtonView

        view = ButtonView(
            cog=cog,
            conversation_starter=conversation_starter,
            conversation_id=111,
        )

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.data = {"values": ["web_search"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction)

        interaction.response.send_message.assert_called_once()
        assert "not allowed" in interaction.response.send_message.call_args.args[0]

    @pytest.mark.asyncio
    async def test_tool_select_callback_shows_resolution_error(
        self, cog, conversation_starter
    ):
        from src.button_view import ButtonView

        view = ButtonView(
            cog=cog,
            conversation_starter=conversation_starter,
            conversation_id=111,
        )

        cog.resolve_selected_tools.return_value = ([], "Collections config missing")

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.data = {"values": ["collections_search"]}
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction)

        interaction.response.send_message.assert_called_once()
        assert (
            "Collections config missing"
            in interaction.response.send_message.call_args.args[0]
        )
