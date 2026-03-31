from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from discord import (
    ButtonStyle,
    Interaction,
    Member,
    SelectOption,
    TextChannel,
    User,
)
from discord.ui import Button, Select, View, button

from .tooling import SELECTABLE_TOOLS, resolve_tool_name


async def _send_interaction_error(interaction: Interaction, context: str, error: Exception) -> None:
    """Log an error and send the user a safe ephemeral message."""
    logging.error(f"Error in {context}: {error}", exc_info=True)
    msg = f"An error occurred while {context}."
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)


class ButtonView(View):
    def __init__(
        self,
        *,
        conversation_starter: Member | User,
        conversation_id: int,
        initial_tools: list[Any] | None = None,
        get_conversation: Callable[[int], Any | None],
        on_regenerate: Callable[[Any, Any], Awaitable[None]],
        on_stop: Callable[[int], Awaitable[None]],
        on_tools_changed: Callable[[list[str], Any], tuple[set[str], str | None]],
    ):
        """
        Initialize the ButtonView class.
        """
        super().__init__(timeout=None)
        self.conversation_starter = conversation_starter
        self.conversation_id = conversation_id
        self._get_conversation = get_conversation
        self._on_regenerate = on_regenerate
        self._on_stop = on_stop
        self._on_tools_changed = on_tools_changed
        self._add_tool_select(initial_tools or [])

    def _add_tool_select(self, initial_tools: list[Any]) -> None:
        selected_tools = {
            tool_name
            for tool in initial_tools
            if (tool_name := resolve_tool_name(tool)) is not None
        }

        tool_select = Select(
            placeholder="Toggle conversation tools",
            min_values=0,
            max_values=len(SELECTABLE_TOOLS),
            row=1,
            options=[
                SelectOption(
                    label="Web Search",
                    value="web_search",
                    description="Search the web in real time.",
                    default="web_search" in selected_tools,
                ),
                SelectOption(
                    label="X Search",
                    value="x_search",
                    description="Search X posts and threads.",
                    default="x_search" in selected_tools,
                ),
                SelectOption(
                    label="Code Execution",
                    value="code_execution",
                    description="Run Python code in a sandbox.",
                    default="code_execution" in selected_tools,
                ),
                SelectOption(
                    label="Collections Search",
                    value="collections_search",
                    description="Search configured collections.",
                    default="collections_search" in selected_tools,
                ),
            ],
        )

        async def _tool_callback(interaction: Interaction) -> None:
            await self.tool_select_callback(interaction, tool_select)

        tool_select.callback = _tool_callback
        self.add_item(tool_select)

    async def tool_select_callback(self, interaction: Interaction, tool_select: Select) -> None:
        """Toggle tool availability for this conversation."""
        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to change tools for this conversation.",
                    ephemeral=True,
                )
                return

            conversation = self._get_conversation(self.conversation_id)
            if conversation is None:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
                return

            selected_values = [value for value in tool_select.values if value in SELECTABLE_TOOLS]

            active_names, error_message = self._on_tools_changed(selected_values, conversation)
            if error_message and not active_names:
                await interaction.response.send_message(error_message, ephemeral=True)
                return

            # Update Select dropdown defaults
            for child in self.children:
                if isinstance(child, Select):
                    for option in child.options:
                        option.default = option.value in active_names
                    break

            if active_names:
                tool_names = ", ".join(sorted(active_names))
                message = f"Tools updated: {tool_names}."
            else:
                message = "Tools disabled for this conversation."

            await interaction.response.send_message(message, ephemeral=True, delete_after=3)
        except Exception as e:
            await _send_interaction_error(interaction, "updating tools", e)

    @button(emoji="🔄", style=ButtonStyle.green, row=0)
    async def regenerate_button(self, _: Button, interaction: Interaction):
        """
        Regenerate the last response for the current conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        logging.info("Regenerate button clicked.")
        saved_response_id: str | None = None
        saved_previous_id: str | None = None

        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to regenerate the response.", ephemeral=True
                )
                return

            conversation = self._get_conversation(self.conversation_id)
            if conversation is None:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
                return

            await interaction.response.defer(ephemeral=True)

            if not conversation.response_id_history:
                await interaction.followup.send(
                    "Not enough history to regenerate yet.", ephemeral=True
                )
                return

            # Save state for rollback
            saved_response_id = conversation.response_id_history[-1]
            saved_previous_id = conversation.previous_response_id

            # Rewind: pop the last response
            conversation.response_id_history.pop()
            conversation.previous_response_id = (
                conversation.response_id_history[-1] if conversation.response_id_history else None
            )

            channel = interaction.channel
            if not hasattr(channel, "history"):
                conversation.response_id_history.append(saved_response_id)
                conversation.previous_response_id = saved_previous_id
                await interaction.followup.send(
                    "Couldn't find the message to regenerate.", ephemeral=True
                )
                return

            text_channel = cast(TextChannel, channel)
            recent = [message async for message in text_channel.history(limit=10)]
            user_message = next(
                (message for message in recent if message.author == self.conversation_starter),
                None,
            )

            if user_message is None:
                conversation.response_id_history.append(saved_response_id)
                conversation.previous_response_id = saved_previous_id
                await interaction.followup.send(
                    "Couldn't find the message to regenerate.", ephemeral=True
                )
                return

            await self._on_regenerate(user_message, conversation)
            await interaction.followup.send("Response regenerated.", ephemeral=True, delete_after=3)
        except Exception as error:
            logging.error(
                f"Error in regenerate_button: {error}",
                exc_info=True,
            )

            if saved_response_id is not None:
                conversation = self._get_conversation(self.conversation_id)
                if conversation is not None:
                    conversation.response_id_history.append(saved_response_id)
                    conversation.previous_response_id = saved_previous_id

            if interaction.response.is_done():
                await interaction.followup.send(
                    "An error occurred while regenerating the response.",
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    "An error occurred while regenerating the response.",
                    ephemeral=True,
                )

    @button(emoji="⏯️", style=ButtonStyle.gray, row=0)
    async def play_pause_button(self, _: Button, interaction: Interaction):
        """
        Pause or resume the conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to pause the conversation.", ephemeral=True
                )
                return

            conversation = self._get_conversation(self.conversation_id)
            if conversation is not None:
                conversation.params.paused = not conversation.params.paused
                status = "paused" if conversation.params.paused else "resumed"
                await interaction.response.send_message(
                    f"Conversation {status}. Press again to toggle.",
                    ephemeral=True,
                    delete_after=3,
                )
            else:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
        except Exception as e:
            await _send_interaction_error(interaction, "toggling pause", e)

    @button(emoji="⏹️", style=ButtonStyle.blurple, row=0)
    async def stop_button(self, button: Button, interaction: Interaction):
        """
        End the conversation.

        Args:
            interaction (Interaction): The interaction object.
        """
        try:
            if interaction.user != self.conversation_starter:
                await interaction.response.send_message(
                    "You are not allowed to end this conversation.", ephemeral=True
                )
                return

            conversation = self._get_conversation(self.conversation_id)
            if conversation is not None:
                await self._on_stop(self.conversation_id)
                button.disabled = True
                await interaction.response.send_message(
                    "Conversation ended.", ephemeral=True, delete_after=3
                )
            else:
                await interaction.response.send_message(
                    "No active conversation found.", ephemeral=True
                )
        except Exception as e:
            await _send_interaction_error(interaction, "ending the conversation", e)
