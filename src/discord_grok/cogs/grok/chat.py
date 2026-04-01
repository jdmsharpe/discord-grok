from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, cast

from discord import ApplicationContext, Attachment, Colour, Embed, Message, TextChannel

from ...config.mcp import (
    build_mcp_server_config,
    parse_mcp_preset_names,
    resolve_mcp_presets,
)
from .attachments import MAX_IMAGE_SIZE, SUPPORTED_IMAGE_TYPES
from .embeds import (
    append_pricing_embed,
    append_reasoning_embeds,
    append_response_embeds,
    append_sources_embed,
)
from .models import ChatCompletionParameters, Conversation
from .state import create_button_view
from .tooling import (
    MULTI_AGENT_MODELS,
    PENALTY_SUPPORTED_MODELS,
    REASONING_EFFORT_MODELS,
    TOOL_CODE_EXECUTION,
    TOOL_COLLECTIONS_SEARCH,
    TOOL_REMOTE_MCP,
    TOOL_WEB_SEARCH,
    TOOL_X_SEARCH,
    calculate_cost,
    calculate_tool_cost,
    format_xai_error,
    truncate_text,
)


async def keep_typing(cog, channel: Any) -> None:
    """Keep the Discord typing indicator alive while Grok is working."""
    try:
        while True:
            async with channel.typing():
                await asyncio.sleep(5)
    except asyncio.CancelledError:
        raise


async def handle_on_message(cog, message: Message) -> None:
    """Route follow-up messages into active Grok conversations."""
    if message.author == cog.bot.user:
        return

    for conversation in cog.conversations.values():
        if message.channel.id != conversation.params.channel_id:
            continue
        if message.author != conversation.params.conversation_starter:
            continue

        cog.logger.info(
            "Processing followup message for conversation %s",
            conversation.params.conversation_id,
        )
        await cog.handle_new_message_in_conversation(message, conversation)
        break


async def handle_check_permissions(ctx: ApplicationContext) -> None:
    """Check whether the bot can read the current server channel."""
    if ctx.guild is None:
        await ctx.respond("This command can only be used in a server.")
        return
    channel = ctx.channel
    if not isinstance(channel, TextChannel):
        await ctx.respond("Cannot check permissions in this channel type.")
        return
    permissions = channel.permissions_for(ctx.guild.me)
    if permissions.read_messages and permissions.read_message_history:
        await ctx.respond("Bot has permission to read messages and message history.")
        return
    await ctx.respond("Bot is missing necessary permissions in this channel.")


async def handle_new_message_in_conversation(cog, message: Message, conversation) -> None:
    """Handle a new Discord message in an ongoing Grok conversation."""
    params = conversation.params
    cog.logger.info("Handling new message in conversation %s.", params.conversation_id)
    typing_task = None
    embeds = []

    try:
        if message.author != params.conversation_starter or params.paused:
            return

        typing_task = asyncio.create_task(keep_typing(cog, message.channel))

        content_parts: list[Any] = []
        if message.content:
            content_parts.append(message.content)
        if message.attachments:
            unsupported_image_error = next(
                (
                    error
                    for attachment in message.attachments
                    if (error := cog._unsupported_image_type_error(cast(Attachment, attachment)))
                ),
                None,
            )
            if unsupported_image_error:
                if typing_task:
                    typing_task.cancel()
                    typing_task = None
                await message.reply(
                    embed=Embed(
                        title="Error",
                        description=unsupported_image_error,
                        color=Colour.red(),
                    )
                )
                return

            for attachment in message.attachments:
                content_type = (attachment.content_type or "").lower()
                if content_type in SUPPORTED_IMAGE_TYPES:
                    if attachment.size > MAX_IMAGE_SIZE:
                        cog.logger.warning(
                            "Image %s exceeds 20 MiB limit (%s bytes), skipping",
                            attachment.filename,
                            attachment.size,
                        )
                        continue
                    content_parts.append(
                        {"type": "input_image", "image_url": attachment.url, "detail": "high"}
                    )
                else:
                    file_id = await cog._upload_file_attachment(attachment)
                    if file_id:
                        conversation.file_ids.append(file_id)
                        content_parts.append({"type": "input_file", "file_id": file_id})

        if not content_parts:
            return

        input_messages = [cog._build_user_message(content_parts)]
        payload = cog._build_responses_payload(
            model=params.model,
            input_messages=input_messages,
            previous_response_id=conversation.previous_response_id,
            prompt_cache_key=conversation.prompt_cache_key,
            tools=params.tools or None,
            max_output_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            reasoning_effort=params.reasoning_effort,
            agent_count=params.agent_count,
            include_encrypted_reasoning=bool(params.tools) or params.model in MULTI_AGENT_MODELS,
        )
        response_json = await cog._call_responses_api(
            payload,
            grok_conv_id=conversation.grok_conv_id,
        )
        response_text, reasoning_text = cog._extract_response_text(response_json)
        tool_info = cog._extract_tool_info(response_json)

        usage = cog._extract_usage(response_json)
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        reasoning_tokens = usage["reasoning_tokens"]
        cached_tokens = usage["cached_tokens"]
        image_tokens = usage["image_tokens"]
        tool_usage = response_json.get("server_side_tool_usage", {})

        if typing_task:
            typing_task.cancel()
            typing_task = None

        response_id = response_json.get("id")
        if response_id:
            conversation.response_id_history.append(response_id)
            conversation.previous_response_id = response_id

        append_reasoning_embeds(embeds, reasoning_text)
        append_response_embeds(embeds, response_text)
        append_sources_embed(embeds, tool_info["citations"])

        request_cost = calculate_cost(
            params.model,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_tokens,
        ) + calculate_tool_cost(tool_usage or {})
        daily_cost = cog._track_daily_cost(message.author.id, request_cost)
        if cog.show_cost_embeds:
            append_pricing_embed(
                embeds,
                request_cost,
                input_tokens,
                output_tokens,
                daily_cost,
                reasoning_tokens,
                cached_tokens,
                image_tokens,
                tool_usage,
            )

        cog._log_chat_cost(
            message.author.id,
            params.model,
            input_tokens,
            cached_tokens,
            output_tokens,
            reasoning_tokens,
            image_tokens,
            tool_usage,
            request_cost,
            daily_cost,
        )

        if params.conversation_id is None:
            cog.logger.error("Conversation ID is None, cannot track message")
            return

        await cog._strip_previous_view(message.author)
        view = cog.views.get(message.author)

        if embeds:
            try:
                reply_message = await message.reply(embeds=embeds, view=view)
                cog.last_view_messages[message.author] = reply_message
            except Exception as embed_error:
                cog.logger.warning("Embed failed, sending as text: %s", embed_error)
                safe_response_text = response_text or "No response text available"
                reply_message = await message.reply(
                    content=(
                        f"**Response:**\n{safe_response_text[:1900]}"
                        f"{'...' if len(safe_response_text) > 1900 else ''}"
                    ),
                    view=view,
                )
                cog.last_view_messages[message.author] = reply_message
            cog.logger.debug("Replied with generated response.")
            return

        cog.logger.warning("No embeds to send in the reply.")
        await message.reply(content="An error occurred: No content to send.")
        if conversation.params.conversation_id is not None:
            await cog.end_conversation(conversation.params.conversation_id)

    except Exception as error:
        description = format_xai_error(error)
        cog.logger.error(
            "Error in handle_new_message_in_conversation: %s",
            description,
            exc_info=True,
        )
        if len(description) > 4000:
            description = description[:4000] + "\n\n... (error message truncated)"
        await message.reply(embed=Embed(title="Error", description=description, color=Colour.red()))
        if conversation.params.conversation_id is not None:
            await cog.end_conversation(conversation.params.conversation_id)

    finally:
        if typing_task:
            typing_task.cancel()


async def run_chat_command(
    cog,
    *,
    ctx: ApplicationContext,
    prompt: str,
    model: str = "grok-4.20",
    system_prompt: str | None = None,
    attachment: Attachment | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: str | None = None,
    agent_count: int | None = None,
    web_search: bool = False,
    x_search: bool = False,
    code_execution: bool = False,
    collections_search: bool = False,
    mcp: str | None = None,
    x_search_images: bool = False,
    x_search_videos: bool = False,
    x_search_date_range: str | None = None,
    web_search_images: bool = False,
) -> None:
    """Create a persistent Grok conversation session."""
    await ctx.defer()
    typing_task = None
    uploaded_file_ids: list[str] = []

    if ctx.channel is None:
        await ctx.send_followup(
            embed=Embed(
                title="Error",
                description="Cannot start conversation: channel context is unavailable.",
                color=Colour.red(),
            )
        )
        return

    for conversation in cog.conversations.values():
        if (
            conversation.params.conversation_starter == ctx.author
            and conversation.params.channel_id == ctx.channel.id
        ):
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=(
                        "You already have an active conversation in this channel. "
                        "Please finish it before starting a new one."
                    ),
                    color=Colour.red(),
                )
            )
            return

    main_conversation_id: int | None = None
    try:
        typing_task = asyncio.create_task(keep_typing(cog, ctx.channel))

        if (
            frequency_penalty is not None or presence_penalty is not None
        ) and model not in PENALTY_SUPPORTED_MODELS:
            unsupported = []
            if frequency_penalty is not None:
                unsupported.append("`frequency_penalty`")
            if presence_penalty is not None:
                unsupported.append("`presence_penalty`")
            param_list = " and ".join(unsupported)
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=(
                        f"{param_list} {'is' if len(unsupported) == 1 else 'are'} "
                        f"not supported by reasoning model `{model}`."
                    ),
                    color=Colour.red(),
                )
            )
            return

        if reasoning_effort is not None and model not in REASONING_EFFORT_MODELS:
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=(
                        "`reasoning_effort` is only supported by "
                        f"{', '.join(f'`{item}`' for item in sorted(REASONING_EFFORT_MODELS))}."
                    ),
                    color=Colour.red(),
                )
            )
            return

        is_multi_agent = model in MULTI_AGENT_MODELS

        if max_tokens is not None and is_multi_agent:
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=f"`max_tokens` is not supported by multi-agent model `{model}`.",
                    color=Colour.red(),
                )
            )
            return

        if agent_count is not None and not is_multi_agent:
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=(
                        "`agent_count` is only supported by multi-agent models "
                        f"({', '.join(f'`{item}`' for item in sorted(MULTI_AGENT_MODELS))})."
                    ),
                    color=Colour.red(),
                )
            )
            return

        x_search_kw: dict[str, Any] = {}
        if x_search_images:
            x_search_kw["enable_image_understanding"] = True
        if x_search_videos:
            x_search_kw["enable_video_understanding"] = True
        if x_search_date_range:
            date_parts = [part.strip() for part in x_search_date_range.split(",")]
            if len(date_parts) != 2 or not all(date_parts):
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=(
                            "Invalid `x_search_date_range` format. Use YYYY-MM-DD,YYYY-MM-DD."
                        ),
                        color=Colour.red(),
                    )
                )
                return
            try:
                x_search_kw["from_date"] = datetime.fromisoformat(date_parts[0]).isoformat()
                x_search_kw["to_date"] = datetime.fromisoformat(date_parts[1]).isoformat()
            except ValueError:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=(
                            "Invalid `x_search_date_range` format. Use YYYY-MM-DD,YYYY-MM-DD."
                        ),
                        color=Colour.red(),
                    )
                )
                return

        web_search_kw: dict[str, Any] = {}
        if web_search_images:
            web_search_kw["enable_image_understanding"] = True

        mcp_preset_names = parse_mcp_preset_names(mcp)
        mcp_presets, mcp_error = resolve_mcp_presets(mcp_preset_names)
        if mcp_error:
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=mcp_error,
                    color=Colour.red(),
                )
            )
            return
        mcp_servers = [build_mcp_server_config(preset) for preset in mcp_presets]

        selected_tool_names: list[str] = []
        if web_search:
            selected_tool_names.append(TOOL_WEB_SEARCH)
        if x_search:
            selected_tool_names.append(TOOL_X_SEARCH)
        if code_execution:
            selected_tool_names.append(TOOL_CODE_EXECUTION)
        if collections_search:
            selected_tool_names.append(TOOL_COLLECTIONS_SEARCH)
        if mcp_servers:
            selected_tool_names.append(TOOL_REMOTE_MCP)

        tools, tool_error = cog.resolve_selected_tools(
            selected_tool_names,
            x_search_kwargs=x_search_kw,
            web_search_kwargs=web_search_kw,
            mcp_servers=mcp_servers,
        )
        if tool_error:
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=tool_error,
                    color=Colour.red(),
                )
            )
            return

        initial_messages: list[dict[str, Any]] = []
        if system_prompt:
            initial_messages.append({"role": "system", "content": system_prompt})

        content_parts: list[Any] = [prompt]
        if attachment:
            unsupported_image_error = cog._unsupported_image_type_error(attachment)
            if unsupported_image_error:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=unsupported_image_error,
                        color=Colour.red(),
                    )
                )
                return

            content_type = (attachment.content_type or "").lower()
            if content_type in SUPPORTED_IMAGE_TYPES:
                if attachment.size > MAX_IMAGE_SIZE:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description=f"Image `{attachment.filename}` exceeds the 20 MiB limit.",
                            color=Colour.red(),
                        ),
                        ephemeral=True,
                    )
                    return
                content_parts.append(
                    {"type": "input_image", "image_url": attachment.url, "detail": "high"}
                )
            else:
                file_id = await cog._upload_file_attachment(attachment)
                if file_id:
                    uploaded_file_ids.append(file_id)
                    content_parts.append({"type": "input_file", "file_id": file_id})
        initial_messages.append(cog._build_user_message(content_parts))

        prompt_cache_key = str(uuid.uuid4())
        grok_conv_id = str(uuid.uuid4())
        payload = cog._build_responses_payload(
            model=model,
            input_messages=initial_messages,
            prompt_cache_key=prompt_cache_key,
            tools=tools or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            agent_count=agent_count if is_multi_agent else None,
            include_encrypted_reasoning=bool(tools) or is_multi_agent,
        )
        response_json = await cog._call_responses_api(payload, grok_conv_id=grok_conv_id)
        response_text, reasoning_text = cog._extract_response_text(response_json)
        tool_info = cog._extract_tool_info(response_json)

        usage = cog._extract_usage(response_json)
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        reasoning_tokens = usage["reasoning_tokens"]
        cached_tokens = usage["cached_tokens"]
        image_tokens = usage["image_tokens"]
        tool_usage = response_json.get("server_side_tool_usage", {})

        truncated_prompt = truncate_text(prompt, 2000)
        description = f"**Prompt:** {truncated_prompt}\n"
        description += f"**Model:** {model}\n"
        if system_prompt:
            description += f"**System:** {truncate_text(system_prompt, 500)}\n"
        if max_tokens is not None:
            description += f"**Max Tokens:** {max_tokens}\n"
        if temperature is not None:
            description += f"**Temperature:** {temperature}\n"
        if top_p is not None:
            description += f"**Top P:** {top_p}\n"
        if frequency_penalty is not None:
            description += f"**Frequency Penalty:** {frequency_penalty}\n"
        if presence_penalty is not None:
            description += f"**Presence Penalty:** {presence_penalty}\n"
        if reasoning_effort is not None:
            description += f"**Reasoning Effort:** {reasoning_effort}\n"
        if agent_count is not None:
            description += f"**Agent Count:** {agent_count}\n"
        if selected_tool_names:
            description += f"**Tools:** {', '.join(selected_tool_names)}\n"
        if mcp_preset_names:
            description += f"**MCP Presets:** {', '.join(mcp_preset_names)}\n"
            if mcp_servers:
                mcp_server = mcp_servers[0]
                description += (
                    f"**MCP Server:** {mcp_server.server_label} ({mcp_server.server_url})\n"
                )
                if mcp_server.allowed_tool_names:
                    description += (
                        "**MCP Allowed Tools:** "
                        + ", ".join(mcp_server.allowed_tool_names)
                        + "\n"
                    )

        embeds = [
            Embed(
                title="Conversation Started",
                description=description,
                color=Colour.green(),
            )
        ]
        append_reasoning_embeds(embeds, reasoning_text)
        append_response_embeds(embeds, response_text)
        append_sources_embed(embeds, tool_info["citations"])

        request_cost = calculate_cost(
            model,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_tokens,
        ) + calculate_tool_cost(tool_usage or {})
        daily_cost = cog._track_daily_cost(ctx.author.id, request_cost)
        if cog.show_cost_embeds:
            append_pricing_embed(
                embeds,
                request_cost,
                input_tokens,
                output_tokens,
                daily_cost,
                reasoning_tokens,
                cached_tokens,
                image_tokens,
                tool_usage,
            )

        cog._log_chat_cost(
            ctx.author.id,
            model,
            input_tokens,
            cached_tokens,
            output_tokens,
            reasoning_tokens,
            image_tokens,
            tool_usage,
            request_cost,
            daily_cost,
        )

        if len(embeds) == 1:
            await ctx.send_followup("No response generated.")
            return

        main_conversation_id = ctx.interaction.id
        await cog._strip_previous_view(ctx.author)
        view = create_button_view(
            cog,
            user=ctx.author,
            conversation_id=main_conversation_id,
            initial_tools=tools,
        )

        message = await ctx.send_followup(embeds=embeds, view=view)
        cog.last_view_messages[ctx.author] = message

        response_id = response_json.get("id")
        params = ChatCompletionParameters(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            agent_count=agent_count,
            tools=tools,
            mcp_servers=mcp_servers,
            x_search_kwargs=x_search_kw,
            web_search_kwargs=web_search_kw,
            conversation_starter=ctx.author,
            channel_id=ctx.channel.id,
            conversation_id=main_conversation_id,
        )
        conversation = Conversation(
            params=params,
            previous_response_id=response_id,
            response_id_history=[response_id] if response_id else [],
            file_ids=uploaded_file_ids,
            prompt_cache_key=prompt_cache_key,
            grok_conv_id=grok_conv_id,
        )
        cog.conversations[main_conversation_id] = conversation

    except Exception as error:
        description = format_xai_error(error)
        cog.logger.error("Error in chat: %s", description, exc_info=True)
        await ctx.send_followup(
            embed=Embed(title="Error", description=description, color=Colour.red())
        )
        if uploaded_file_ids:
            client = cog._get_client()
            for file_id in uploaded_file_ids:
                try:
                    await client.files.delete(file_id)
                    cog.logger.info("Cleaned up orphaned xAI file %s", file_id)
                except Exception as cleanup_error:
                    cog.logger.warning(
                        "Failed to clean up orphaned xAI file %s: %s",
                        file_id,
                        cleanup_error,
                    )
        await cog._strip_previous_view(ctx.author)
        cog.views.pop(ctx.author, None)
        if main_conversation_id is not None:
            cog.conversations.pop(main_conversation_id, None)

    finally:
        if typing_task:
            typing_task.cancel()


__all__ = [
    "handle_check_permissions",
    "handle_new_message_in_conversation",
    "handle_on_message",
    "keep_typing",
    "run_chat_command",
]
