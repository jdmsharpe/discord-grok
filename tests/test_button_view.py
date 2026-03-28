from unittest.mock import AsyncMock, MagicMock

import pytest
from discord.ui import Select

from util import TOOL_BUILDERS


def _make_view(
    conversation_starter=None,
    conversation_id=None,
    initial_tools=None,
    get_conversation=None,
    on_tools_changed=None,
    on_stop=None,
):
    from button_view import ButtonView

    return ButtonView(
        conversation_starter=conversation_starter or MagicMock(),
        conversation_id=conversation_id or 111,
        initial_tools=initial_tools,
        get_conversation=get_conversation or MagicMock(return_value=None),
        on_regenerate=AsyncMock(),
        on_stop=on_stop or AsyncMock(),
        on_tools_changed=on_tools_changed or MagicMock(return_value=(set(), None)),
    )


class TestButtonView:
    @pytest.fixture
    def conversation_starter(self):
        user = MagicMock()
        user.id = 12345
        return user

    async def test_init_adds_tool_select(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)

        selects = [item for item in view.children if isinstance(item, Select)]
        assert len(selects) == 1
        assert selects[0].min_values == 0
        assert selects[0].max_values == 4

    async def test_init_with_initial_tools_sets_defaults(self, conversation_starter):
        view = _make_view(
            conversation_starter=conversation_starter,
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

    async def test_tool_select_callback_updates_tools(self, conversation_starter):
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []

        active = {"web_search", "code_execution"}
        on_tools_changed = MagicMock(return_value=(active, None))

        view = _make_view(
            conversation_starter=conversation_starter,
            get_conversation=MagicMock(return_value=conversation),
            on_tools_changed=on_tools_changed,
        )

        # Get real select for verifying defaults; use mock for .values
        real_select = next(item for item in view.children if isinstance(item, Select))
        mock_select = MagicMock()
        mock_select.values = ["web_search", "code_execution"]

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.tool_select_callback(interaction, mock_select)

        on_tools_changed.assert_called_once_with(
            ["web_search", "code_execution"], conversation
        )

        call_args = interaction.response.send_message.call_args
        assert "Tools updated" in call_args.args[0]
        assert call_args.kwargs["ephemeral"] is True

        # Verify Select defaults were updated on the real widget
        defaults = {option.value: option.default for option in real_select.options}
        assert defaults["web_search"] is True
        assert defaults["code_execution"] is True
        assert defaults["x_search"] is False

    async def test_tool_select_callback_rejects_non_owner(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction, mock_select)

        interaction.response.send_message.assert_called_once()
        assert "not allowed" in interaction.response.send_message.call_args.args[0]

    async def test_tool_select_callback_shows_resolution_error(
        self, conversation_starter
    ):
        conversation = MagicMock()
        on_tools_changed = MagicMock(
            return_value=(set(), "Collections config missing")
        )

        view = _make_view(
            conversation_starter=conversation_starter,
            get_conversation=MagicMock(return_value=conversation),
            on_tools_changed=on_tools_changed,
        )

        mock_select = MagicMock()
        mock_select.values = ["collections_search"]

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction, mock_select)

        interaction.response.send_message.assert_called_once()
        assert (
            "Collections config missing"
            in interaction.response.send_message.call_args.args[0]
        )

    async def test_stop_button_calls_on_stop(self, conversation_starter):
        conversation = MagicMock()
        on_stop = AsyncMock()

        view = _make_view(
            conversation_starter=conversation_starter,
            conversation_id=111,
            get_conversation=MagicMock(return_value=conversation),
            on_stop=on_stop,
        )

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.stop_button.callback(interaction)

        on_stop.assert_awaited_once_with(111)
        call_args = interaction.response.send_message.call_args
        assert "Conversation ended" in call_args.args[0]

    async def test_stop_button_rejects_non_owner(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.stop_button.callback(interaction)

        call_args = interaction.response.send_message.call_args
        assert "not allowed" in call_args.args[0]

    async def test_stop_button_no_conversation(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.stop_button.callback(interaction)

        call_args = interaction.response.send_message.call_args
        assert "No active conversation" in call_args.args[0]

    async def test_play_pause_toggles(self, conversation_starter):
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.paused = False

        view = _make_view(
            conversation_starter=conversation_starter,
            get_conversation=MagicMock(return_value=conversation),
        )

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.play_pause_button.callback(interaction)
        assert conversation.params.paused is True
        assert "paused" in interaction.response.send_message.call_args.args[0]

        interaction.response.send_message.reset_mock()
        await view.play_pause_button.callback(interaction)
        assert conversation.params.paused is False
        assert "resumed" in interaction.response.send_message.call_args.args[0]

    async def test_play_pause_rejects_non_owner(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.play_pause_button.callback(interaction)

        call_args = interaction.response.send_message.call_args
        assert "not allowed" in call_args.args[0]

    async def test_play_pause_no_conversation(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.play_pause_button.callback(interaction)

        call_args = interaction.response.send_message.call_args
        assert "No active conversation" in call_args.args[0]

    async def test_regenerate_rejects_non_owner(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.regenerate_button.callback(interaction)

        call_args = interaction.response.send_message.call_args
        assert "not allowed" in call_args.args[0]

    async def test_regenerate_no_conversation(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.regenerate_button.callback(interaction)

        call_args = interaction.response.send_message.call_args
        assert "No active conversation" in call_args.args[0]

    async def test_regenerate_no_history(self, conversation_starter):
        conversation = MagicMock()
        conversation.response_id_history = []

        view = _make_view(
            conversation_starter=conversation_starter,
            get_conversation=MagicMock(return_value=conversation),
        )

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.is_done = MagicMock(return_value=False)
        interaction.response.defer = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        await view.regenerate_button.callback(interaction)

        call_args = interaction.followup.send.call_args
        assert "Not enough history" in call_args.args[0]

    async def test_tool_select_no_conversation(self, conversation_starter):
        view = _make_view(conversation_starter=conversation_starter)
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = conversation_starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction, mock_select)

        call_args = interaction.response.send_message.call_args
        assert "No active conversation" in call_args.args[0]
