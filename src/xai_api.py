import asyncio
import io
import logging
from datetime import date, datetime
from typing import Any, TypedDict, cast

import aiohttp

from xai_sdk import AsyncClient
from xai_sdk.chat import file as xai_file, image, system, user
from xai_sdk.image import ImageAspectRatio
from xai_sdk.tools import collections_search as collections_search_tool
from xai_sdk.video import VideoAspectRatio, VideoResolution

from discord import (
    ApplicationContext,
    Attachment,
    Colour,
    Embed,
    File,
)
from discord.commands import OptionChoice, SlashCommandGroup, option
from discord.ext import commands

from button_view import ButtonView
from config.auth import GUILD_IDS, SHOW_COST_EMBEDS, XAI_API_KEY, XAI_COLLECTION_IDS
from util import (
    ChatCompletionParameters,
    Conversation,
    PENALTY_SUPPORTED_MODELS,
    REASONING_EFFORT_MODELS,
    TOOL_BUILDERS,
    TOOL_COLLECTIONS_SEARCH,
    TOOL_WEB_SEARCH,
    TOOL_X_SEARCH,
    calculate_cost,
    calculate_image_cost,
    calculate_video_cost,
    chunk_text,
    format_xai_error,
    truncate_text,
)

TTS_API_URL = "https://api.x.ai/v1/tts"
TTS_MAX_CHARS = 15_000

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_FILE_SIZE = 48 * 1024 * 1024  # 48 MB xAI Files API limit
INLINE_CITATION_INCLUDE = "inline_citations"


class ToolInfo(TypedDict):
    citations: list[str]


def append_reasoning_embeds(embeds: list[Embed], reasoning_text: str) -> None:
    """Append reasoning text as a spoilered Discord embed."""
    if not reasoning_text:
        return

    if len(reasoning_text) > 3500:
        reasoning_text = reasoning_text[:3450] + "\n\n... [reasoning truncated]"

    embeds.append(
        Embed(
            title="Reasoning",
            description=f"||{reasoning_text}||",
            color=Colour.light_grey(),
        )
    )


def append_response_embeds(embeds: list[Embed], response_text: str) -> None:
    """Append response text as Discord embeds, handling chunking for long responses."""
    if len(response_text) > 20000:
        response_text = (
            response_text[:19500] + "\n\n... [Response truncated due to length]"
        )

    for index, chunk in enumerate(chunk_text(response_text, 3500), start=1):
        embeds.append(
            Embed(
                title="Response" + (f" (Part {index})" if index > 1 else ""),
                description=chunk,
                color=Colour.dark_teal(),
            )
        )


def extract_tool_info(response: Any) -> ToolInfo:
    """Extract citation links from an xAI SDK response."""
    citations: list[str] = []
    seen_citations: set[str] = set()

    citations_value = getattr(response, "citations", [])
    if citations_value is None:
        citation_items: list[Any] = []
    elif isinstance(citations_value, (list, tuple, set)):
        citation_items = list(citations_value)
    else:
        try:
            citation_items = list(citations_value)
        except TypeError:
            citation_items = []

    for citation in citation_items:
        citation_url = str(citation).strip()
        if not citation_url or citation_url in seen_citations:
            continue
        seen_citations.add(citation_url)
        citations.append(citation_url)

    return {"citations": citations}


def append_sources_embed(embeds: list[Embed], citations: list[str]) -> None:
    """Append a compact sources embed for tool-backed responses."""
    if not citations or len(embeds) >= 10:
        return

    source_lines: list[str] = []
    for index, citation_url in enumerate(citations[:8], start=1):
        if citation_url.startswith("http://") or citation_url.startswith("https://"):
            citation_title = truncate_text(
                citation_url.removeprefix("https://").removeprefix("http://"),
                120,
            )
            source_lines.append(f"{index}. [{citation_title}]({citation_url})")
        else:
            source_lines.append(f"{index}. `{truncate_text(citation_url, 300)}`")

    description = "\n".join(source_lines)
    if len(description) > 4000:
        description = truncate_text(description, 3990)

    embeds.append(
        Embed(
            title="Sources",
            description=description,
            color=Colour.dark_teal(),
        )
    )


def append_pricing_embed(
    embeds: list[Embed],
    model: str,
    input_tokens: int,
    output_tokens: int,
    daily_cost: float,
    reasoning_tokens: int = 0,
) -> None:
    """Append a compact pricing embed showing cost and token usage."""
    if not SHOW_COST_EMBEDS:
        return
    cost = calculate_cost(model, input_tokens, output_tokens)
    token_info = f"{input_tokens:,} in / {output_tokens:,} out"
    if reasoning_tokens > 0:
        token_info += f" ({reasoning_tokens:,} reasoning)"
    description = f"{model} · ${cost:.4f} · {token_info} · daily ${daily_cost:.2f}"
    embeds.append(Embed(description=description, color=Colour.dark_teal()))


def append_generation_pricing_embed(
    embeds: list[Embed],
    cost: float,
    daily_cost: float,
) -> None:
    """Append a compact pricing embed for image/video generation."""
    if not SHOW_COST_EMBEDS:
        return
    description = f"${cost:.4f} · daily ${daily_cost:.2f}"
    embeds.append(Embed(description=description, color=Colour.dark_teal()))


class xAIAPI(commands.Cog):
    grok = SlashCommandGroup("grok", "xAI Grok commands", guild_ids=GUILD_IDS)

    def __init__(self, bot):
        """
        Initialize the xAIAPI class.

        Args:
            bot: The bot instance.
        """
        self.bot = bot
        self.client = AsyncClient(api_key=XAI_API_KEY)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Dictionary to store conversation state for each chat interaction
        self.conversations: dict[int, Conversation] = {}
        # Dictionary to map any message ID to the main conversation ID for tracking
        self.message_to_conversation_id: dict[int, int] = {}
        # Dictionary to store UI views for each conversation
        self.views = {}
        # Daily cost tracking: (user_id, date_iso) -> cumulative cost
        self.daily_costs: dict[tuple[int, str], float] = {}
        self._http_session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session and not self._http_session.closed:
            return self._http_session
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession()
            return self._http_session

    def _track_daily_cost(
        self, user_id: int, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Add this request's cost to the user's daily total and return the new daily total."""
        cost = calculate_cost(model, input_tokens, output_tokens)
        key = (user_id, date.today().isoformat())
        self.daily_costs[key] = self.daily_costs.get(key, 0.0) + cost
        return self.daily_costs[key]

    def _track_daily_cost_flat(self, user_id: int, cost: float) -> float:
        """Add a flat cost to the user's daily total and return the new daily total."""
        key = (user_id, date.today().isoformat())
        self.daily_costs[key] = self.daily_costs.get(key, 0.0) + cost
        return self.daily_costs[key]

    async def _generate_tts(
        self, text: str, voice_id: str, language: str, codec: str
    ) -> bytes:
        """Call the xAI TTS REST endpoint and return raw audio bytes."""
        session = await self._get_http_session()
        payload: dict[str, Any] = {
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "output_format": {"codec": codec},
        }
        async with session.post(
            TTS_API_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                raise Exception(
                    f"TTS API error (HTTP {resp.status}): {error_body}"
                )
            return await resp.read()

    async def _fetch_attachment_bytes(self, attachment: Attachment) -> bytes | None:
        session = await self._get_http_session()
        try:
            async with session.get(attachment.url) as response:
                if response.status == 200:
                    return await response.read()
                self.logger.warning(
                    "Failed to fetch attachment %s: HTTP %s",
                    attachment.url,
                    response.status,
                )
        except aiohttp.ClientError as error:
            self.logger.warning(
                "Error fetching attachment %s: %s", attachment.url, error
            )
        return None

    async def _upload_file_attachment(self, attachment: Attachment) -> str | None:
        """Download a Discord attachment and upload it to the xAI Files API.

        Returns the xAI file ID, or None on failure.
        """
        if attachment.size > MAX_FILE_SIZE:
            self.logger.warning(
                "Attachment %s exceeds 48 MB limit (%s bytes)",
                attachment.filename,
                attachment.size,
            )
            return None

        file_bytes = await self._fetch_attachment_bytes(attachment)
        if file_bytes is None:
            return None

        try:
            uploaded = await self.client.files.upload(
                file_bytes, filename=attachment.filename
            )
            self.logger.info(
                "Uploaded file %s as %s", attachment.filename, uploaded.id
            )
            return uploaded.id
        except Exception as error:
            self.logger.warning(
                "Failed to upload file %s to xAI: %s", attachment.filename, error
            )
            return None

    async def _cleanup_conversation_files(self, conversation: Conversation) -> None:
        """Delete all xAI files associated with a conversation."""
        for file_id in conversation.file_ids:
            try:
                await self.client.files.delete(file_id)
                self.logger.info("Deleted xAI file %s", file_id)
            except Exception as error:
                self.logger.warning(
                    "Failed to delete xAI file %s: %s", file_id, error
                )
        conversation.file_ids.clear()

    async def end_conversation(self, conversation_id: int) -> None:
        """End a conversation and clean up associated resources."""
        conversation = self.conversations.pop(conversation_id, None)
        if conversation is not None:
            await self._cleanup_conversation_files(conversation)

    def cog_unload(self):
        loop = getattr(self.bot, "loop", None)

        session = self._http_session
        if session and not session.closed:
            if loop and loop.is_running():
                loop.create_task(session.close())
            else:
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(session.close())
                finally:
                    new_loop.close()
        self._http_session = None

    def resolve_selected_tools(
        self,
        selected_tool_names: list[str],
        x_search_kwargs: dict[str, Any] | None = None,
        web_search_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[Any], str | None]:
        """Build tool payloads for the selected tool names."""
        tools: list[Any] = []

        for tool_name in selected_tool_names:
            if tool_name == TOOL_COLLECTIONS_SEARCH:
                if not XAI_COLLECTION_IDS:
                    return (
                        [],
                        "Collections search requires XAI_COLLECTION_IDS to be set in your .env.",
                    )
                tools.append(
                    collections_search_tool(collection_ids=XAI_COLLECTION_IDS.copy())
                )
                continue

            if tool_name == TOOL_X_SEARCH and x_search_kwargs:
                tool_builder = TOOL_BUILDERS.get(tool_name)
                if tool_builder is not None:
                    tools.append(tool_builder(**x_search_kwargs))
                continue

            if tool_name == TOOL_WEB_SEARCH and web_search_kwargs:
                tool_builder = TOOL_BUILDERS.get(tool_name)
                if tool_builder is not None:
                    tools.append(tool_builder(**web_search_kwargs))
                continue

            tool_builder = TOOL_BUILDERS.get(tool_name)
            if tool_builder is None:
                continue
            tools.append(tool_builder())

        return tools, None

    def _apply_tools_to_chat(self, chat: Any, tools: list[Any]) -> None:
        """Apply the current tool set to a mutable xAI chat request."""
        chat_proto = getattr(chat, "proto", None)
        if chat_proto is None:
            return

        try:
            chat_proto.ClearField("tools")
            if tools:
                chat_proto.tools.extend(tools)
        except Exception as error:
            self.logger.warning("Unable to update chat tools dynamically: %s", error)

    async def handle_new_message_in_conversation(
        self, message, conversation: Conversation
    ):
        """
        Handles a new message in an ongoing conversation.

        Args:
            message: The incoming Discord Message object.
            conversation: The conversation object.
        """
        params = conversation.params
        chat = conversation.chat

        self.logger.info(
            f"Handling new message in conversation {params.conversation_id}."
        )
        typing_task = None
        embeds = []

        try:
            if message.author != params.conversation_starter or params.paused:
                return

            typing_task = asyncio.create_task(self.keep_typing(message.channel))

            # Build user message content
            content_parts = []
            if message.content:
                content_parts.append(message.content)
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type in SUPPORTED_IMAGE_TYPES:
                        content_parts.append(image(attachment.url))
                    else:
                        file_id = await self._upload_file_attachment(attachment)
                        if file_id:
                            conversation.file_ids.append(file_id)
                            content_parts.append(xai_file(file_id))

            if content_parts:
                chat.append(user(*content_parts))

            self._apply_tools_to_chat(chat, params.tools)
            response = await chat.sample()
            response_text = response.content or "No response."
            reasoning_text = response.reasoning_content or ""
            tool_info = extract_tool_info(response)

            # Extract token usage (xAI SDK uses prompt_tokens/completion_tokens)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0

            # Stop typing as soon as we have the response
            if typing_task:
                typing_task.cancel()
                typing_task = None

            # Add assistant response to chat history
            chat.append(response)

            append_reasoning_embeds(embeds, reasoning_text)
            append_response_embeds(embeds, response_text)
            append_sources_embed(embeds, tool_info["citations"])
            daily_cost = self._track_daily_cost(
                message.author.id, params.model, input_tokens, output_tokens
            )
            append_pricing_embed(
                embeds, params.model, input_tokens, output_tokens, daily_cost, reasoning_tokens
            )

            view = self.views.get(message.author)
            main_conversation_id = params.conversation_id

            if main_conversation_id is None:
                self.logger.error("Conversation ID is None, cannot track message")
                return

            if embeds:
                try:
                    reply_message = await message.reply(embed=embeds[0], view=view)
                    self.message_to_conversation_id[reply_message.id] = (
                        main_conversation_id
                    )
                except Exception as embed_error:
                    self.logger.warning(f"Embed failed, sending as text: {embed_error}")
                    safe_response_text = response_text or "No response text available"
                    reply_message = await message.reply(
                        content=f"**Response:**\n{safe_response_text[:1900]}{'...' if len(safe_response_text) > 1900 else ''}",
                        view=view,
                    )
                    self.message_to_conversation_id[reply_message.id] = (
                        main_conversation_id
                    )

                for embed in embeds[1:]:
                    try:
                        followup_message = await message.channel.send(
                            embed=embed, view=view
                        )
                        self.message_to_conversation_id[followup_message.id] = (
                            main_conversation_id
                        )
                    except Exception as embed_error:
                        self.logger.warning(f"Followup embed failed: {embed_error}")
                        followup_message = await message.channel.send(
                            content=f"**Response (continued):**\n{embed.description[:1900]}{'...' if len(embed.description) > 1900 else ''}",
                            view=view,
                        )
                        self.message_to_conversation_id[followup_message.id] = (
                            main_conversation_id
                        )

                self.logger.debug("Replied with generated response.")
            else:
                self.logger.warning("No embeds to send in the reply.")
                await message.reply(
                    content="An error occurred: No content to send.",
                    view=view,
                )

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(
                f"Error in handle_new_message_in_conversation: {description}",
                exc_info=True,
            )
            if len(description) > 4000:
                description = description[:4000] + "\n\n... (error message truncated)"
            await message.reply(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

        finally:
            if typing_task:
                typing_task.cancel()

    async def keep_typing(self, channel):
        """
        Coroutine to keep the typing indicator alive in a channel.

        Args:
            channel: The Discord channel object.
        """
        try:
            while True:
                async with channel.typing():
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise

    @commands.Cog.listener()
    async def on_ready(self):
        """
        Event listener that runs when the bot is ready.
        """
        self.logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.owner_id})")
        self.logger.info(f"Attempting to sync commands for guilds: {GUILD_IDS}")
        try:
            await self.bot.sync_commands()
            self.logger.info("Commands synchronized successfully.")
        except Exception as e:
            self.logger.error(
                f"Error during command synchronization: {e}", exc_info=True
            )

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Event listener that runs when a message is sent.
        Generates a response using Grok when a new message from the conversation author is detected.

        Args:
            message: The incoming Discord Message object.
        """
        if message.author == self.bot.user:
            return

        for conversation in self.conversations.values():
            if message.channel.id != conversation.params.channel_id:
                continue
            if message.author != conversation.params.conversation_starter:
                continue

            self.logger.info(
                f"Processing followup message for conversation {conversation.params.conversation_id}"
            )
            await self.handle_new_message_in_conversation(message, conversation)
            break

    @commands.Cog.listener()
    async def on_error(self, event, *args, **kwargs):
        """
        Event listener that runs when an error occurs.
        """
        self.logger.error(f"Error in event {event}: {args} {kwargs}", exc_info=True)

    @grok.command(
        name="check_permissions",
        description="Check if bot has necessary permissions in this channel",
    )
    async def check_permissions(self, ctx: ApplicationContext):
        """
        Checks and reports the bot's permissions in the current channel.
        """
        permissions = ctx.channel.permissions_for(ctx.guild.me)
        if permissions.read_messages and permissions.read_message_history:
            await ctx.respond(
                "Bot has permission to read messages and message history."
            )
        else:
            await ctx.respond("Bot is missing necessary permissions in this channel.")

    @grok.command(
        name="chat",
        description="Starts a conversation with Grok.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "system_prompt",
        description="System prompt to set Grok's behavior. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "model",
        description="Choose from the following Grok models. (default: Grok 4.20 Beta Reasoning)",
        required=False,
        choices=[
            OptionChoice(name="Grok 4.20 Multi-Agent Beta", value="grok-4.20-multi-agent-beta-latest"),
            OptionChoice(name="Grok 4.20 Beta Reasoning", value="grok-4.20-beta-latest-reasoning"),
            OptionChoice(name="Grok 4.20 Beta Non-Reasoning", value="grok-4.20-beta-latest-non-reasoning"),
            OptionChoice(name="Grok 4.1 Fast Reasoning", value="grok-4-1-fast-reasoning"),
            OptionChoice(name="Grok 4.1 Fast Non-Reasoning", value="grok-4-1-fast-non-reasoning"),
            OptionChoice(name="Grok Code Fast 1", value="grok-code-fast-1"),
            OptionChoice(name="Grok 4 Fast Reasoning", value="grok-4-fast-reasoning"),
            OptionChoice(name="Grok 4 Fast Non-Reasoning", value="grok-4-fast-non-reasoning"),
            OptionChoice(name="Grok 4 (0709)", value="grok-4-0709"),
            OptionChoice(name="Grok 3 Mini", value="grok-3-mini"),
            OptionChoice(name="Grok 3", value="grok-3"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="Attach an image or document (PDF, TXT, CSV, code files, etc.).",
        required=False,
        type=Attachment,
    )
    @option(
        "max_tokens",
        description="Maximum tokens in the response. (default: not set)",
        required=False,
        type=int,
    )
    @option(
        "temperature",
        description="(Advanced) Controls the randomness of the model. 0.0 to 2.0. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "top_p",
        description="(Advanced) Nucleus sampling. 0.0 to 1.0. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "frequency_penalty",
        description="(Advanced) Repetition control. Non-reasoning models only. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "presence_penalty",
        description="(Advanced) New topic control. Non-reasoning models only. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "reasoning_effort",
        description="(Advanced) How hard the model thinks. grok-3-mini only. (default: not set)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="Low", value="low"),
            OptionChoice(name="High", value="high"),
        ],
    )
    @option(
        "web_search",
        description="Enable web search for real-time web results. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "x_search",
        description="Enable X search for posts and threads. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "code_execution",
        description="Enable code execution for calculations and analysis. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "collections_search",
        description="Enable collections search over XAI_COLLECTION_IDS. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "x_search_images",
        description="Allow X search to analyze images in posts. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "x_search_videos",
        description="Allow X search to analyze videos in posts. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "x_search_date_range",
        description="X search date range as YYYY-MM-DD,YYYY-MM-DD (start,end). (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "x_search_allowed_handles",
        description="Only search posts from these X handles, comma-separated, max 10. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "x_search_excluded_handles",
        description="Exclude posts from these X handles, comma-separated, max 10. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "web_search_allowed_domains",
        description="Only search these domains, comma-separated, max 5. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "web_search_excluded_domains",
        description="Exclude these domains from search, comma-separated, max 5. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "web_search_images",
        description="Allow web search to analyze images on web pages. (default: false)",
        required=False,
        type=bool,
    )
    async def chat(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "grok-4.20-beta-latest-reasoning",
        system_prompt: str | None = None,
        attachment: Attachment | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        reasoning_effort: str | None = None,
        web_search: bool = False,
        x_search: bool = False,
        code_execution: bool = False,
        collections_search: bool = False,
        x_search_images: bool = False,
        x_search_videos: bool = False,
        x_search_date_range: str | None = None,
        x_search_allowed_handles: str | None = None,
        x_search_excluded_handles: str | None = None,
        web_search_allowed_domains: str | None = None,
        web_search_excluded_domains: str | None = None,
        web_search_images: bool = False,
    ):
        """
        Creates a persistent conversation session with Grok.

        Initiates an interactive conversation with context preservation across multiple exchanges.
        Supports multimodal inputs (text + images) and provides interactive UI controls for
        conversation management.
        """
        await ctx.defer()
        typing_task = None

        if ctx.channel is None:
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description="Cannot start conversation: channel context is unavailable.",
                    color=Colour.red(),
                )
            )
            return

        for conv in self.conversations.values():
            if (
                conv.params.conversation_starter == ctx.author
                and conv.params.channel_id == ctx.channel.id
            ):
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description="You already have an active conversation in this channel. Please finish it before starting a new one.",
                        color=Colour.red(),
                    )
                )
                return

        try:
            typing_task = asyncio.create_task(self.keep_typing(ctx.channel))

            # Validate reasoning model parameter constraints
            if (frequency_penalty is not None or presence_penalty is not None) and model not in PENALTY_SUPPORTED_MODELS:
                unsupported = []
                if frequency_penalty is not None:
                    unsupported.append("`frequency_penalty`")
                if presence_penalty is not None:
                    unsupported.append("`presence_penalty`")
                param_list = " and ".join(unsupported)
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=f"{param_list} {'is' if len(unsupported) == 1 else 'are'} not supported by reasoning model `{model}`.",
                        color=Colour.red(),
                    )
                )
                return

            if reasoning_effort is not None and model not in REASONING_EFFORT_MODELS:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=f"`reasoning_effort` is only supported by {', '.join(f'`{m}`' for m in sorted(REASONING_EFFORT_MODELS))}.",
                        color=Colour.red(),
                    )
                )
                return

            # Build x_search kwargs from optional parameters
            x_search_kw: dict[str, Any] = {}
            if x_search_images:
                x_search_kw["enable_image_understanding"] = True
            if x_search_videos:
                x_search_kw["enable_video_understanding"] = True
            if x_search_date_range:
                date_parts = [p.strip() for p in x_search_date_range.split(",")]
                if len(date_parts) != 2 or not all(date_parts):
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description="Invalid `x_search_date_range` format. Use YYYY-MM-DD,YYYY-MM-DD.",
                            color=Colour.red(),
                        )
                    )
                    return
                try:
                    x_search_kw["from_date"] = datetime.fromisoformat(date_parts[0])
                    x_search_kw["to_date"] = datetime.fromisoformat(date_parts[1])
                except ValueError:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description="Invalid `x_search_date_range` format. Use YYYY-MM-DD,YYYY-MM-DD.",
                            color=Colour.red(),
                        )
                    )
                    return
            if x_search_allowed_handles and x_search_excluded_handles:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description="Cannot use both `x_search_allowed_handles` and `x_search_excluded_handles` at the same time.",
                        color=Colour.red(),
                    )
                )
                return
            if x_search_allowed_handles:
                handles = [h.strip().lstrip("@") for h in x_search_allowed_handles.split(",") if h.strip()]
                if len(handles) > 10:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description="`x_search_allowed_handles` supports a maximum of 10 handles.",
                            color=Colour.red(),
                        )
                    )
                    return
                x_search_kw["allowed_x_handles"] = handles
            if x_search_excluded_handles:
                handles = [h.strip().lstrip("@") for h in x_search_excluded_handles.split(",") if h.strip()]
                if len(handles) > 10:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description="`x_search_excluded_handles` supports a maximum of 10 handles.",
                            color=Colour.red(),
                        )
                    )
                    return
                x_search_kw["excluded_x_handles"] = handles

            # Build web_search kwargs from optional parameters
            web_search_kw: dict[str, Any] = {}
            if web_search_images:
                web_search_kw["enable_image_understanding"] = True
            if web_search_allowed_domains and web_search_excluded_domains:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description="Cannot use both `web_search_allowed_domains` and `web_search_excluded_domains` at the same time.",
                        color=Colour.red(),
                    )
                )
                return
            if web_search_allowed_domains:
                domains = [d.strip() for d in web_search_allowed_domains.split(",") if d.strip()]
                if len(domains) > 5:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description="`web_search_allowed_domains` supports a maximum of 5 domains.",
                            color=Colour.red(),
                        )
                    )
                    return
                web_search_kw["allowed_domains"] = domains
            if web_search_excluded_domains:
                domains = [d.strip() for d in web_search_excluded_domains.split(",") if d.strip()]
                if len(domains) > 5:
                    await ctx.send_followup(
                        embed=Embed(
                            title="Error",
                            description="`web_search_excluded_domains` supports a maximum of 5 domains.",
                            color=Colour.red(),
                        )
                    )
                    return
                web_search_kw["excluded_domains"] = domains

            selected_tool_names: list[str] = []
            if web_search:
                selected_tool_names.append("web_search")
            if x_search:
                selected_tool_names.append("x_search")
            if code_execution:
                selected_tool_names.append("code_execution")
            if collections_search:
                selected_tool_names.append("collections_search")

            tools, tool_error = self.resolve_selected_tools(
                selected_tool_names,
                x_search_kwargs=x_search_kw,
                web_search_kwargs=web_search_kw,
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

            # Build initial messages
            initial_messages = []
            if system_prompt:
                initial_messages.append(system(system_prompt))

            # Build user message content parts
            uploaded_file_ids: list[str] = []
            content_parts: list[Any] = [prompt]
            if attachment:
                if attachment.content_type in SUPPORTED_IMAGE_TYPES:
                    content_parts.append(image(attachment.url))
                else:
                    file_id = await self._upload_file_attachment(attachment)
                    if file_id:
                        uploaded_file_ids.append(file_id)
                        content_parts.append(xai_file(file_id))
            initial_messages.append(user(*content_parts))

            # Build create kwargs
            create_kwargs = {
                "model": model,
                "messages": initial_messages,
                "include": [INLINE_CITATION_INCLUDE],
            }
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            if top_p is not None:
                create_kwargs["top_p"] = top_p
            if frequency_penalty is not None:
                create_kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                create_kwargs["presence_penalty"] = presence_penalty
            if reasoning_effort is not None:
                create_kwargs["reasoning_effort"] = reasoning_effort
            if tools:
                create_kwargs["tools"] = tools
            chat = self.client.chat.create(**create_kwargs)
            response = await chat.sample()
            response_text = response.content or "No response."
            reasoning_text = response.reasoning_content or ""
            tool_info = extract_tool_info(response)

            # Extract token usage (xAI SDK uses prompt_tokens/completion_tokens)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0

            # Add assistant response to chat history
            chat.append(response)

            # Build description embed
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
            if selected_tool_names:
                description += f"**Tools:** {', '.join(selected_tool_names)}\n"

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
            daily_cost = self._track_daily_cost(
                ctx.author.id, model, input_tokens, output_tokens
            )
            append_pricing_embed(
                embeds, model, input_tokens, output_tokens, daily_cost, reasoning_tokens
            )

            if len(embeds) == 1:
                await ctx.send_followup("No response generated.")
                return

            # Create the view with buttons
            main_conversation_id = ctx.interaction.id
            view = ButtonView(
                cog=self,
                conversation_starter=ctx.author,
                conversation_id=main_conversation_id,
                initial_tools=tools,
            )
            self.views[ctx.author] = view

            msg = await ctx.send_followup(embeds=embeds, view=view)
            self.message_to_conversation_id[msg.id] = main_conversation_id

            # Store the conversation
            params = ChatCompletionParameters(
                model=model,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                tools=tools,
                x_search_kwargs=x_search_kw,
                web_search_kwargs=web_search_kw,
                conversation_starter=ctx.author,
                channel_id=ctx.channel.id,
                conversation_id=main_conversation_id,
            )
            conversation = Conversation(
                params=params, chat=chat, file_ids=uploaded_file_ids
            )
            self.conversations[main_conversation_id] = conversation

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(
                f"Error in chat: {description}",
                exc_info=True,
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

        finally:
            if typing_task:
                typing_task.cancel()

    @grok.command(
        name="image",
        description="Generates an image from a prompt.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "model",
        description="Choose from the following image generation models. (default: Grok Imagine Image Pro)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="Grok Imagine Image Pro", value="grok-imagine-image-pro"),
            OptionChoice(name="Grok Imagine Image", value="grok-imagine-image"),
        ],
    )
    @option(
        "aspect_ratio",
        description="Aspect ratio of the image. (default: 1:1)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="1:1 (Square)", value="1:1"),
            OptionChoice(name="16:9 (Landscape)", value="16:9"),
            OptionChoice(name="9:16 (Portrait)", value="9:16"),
            OptionChoice(name="4:3", value="4:3"),
            OptionChoice(name="3:4", value="3:4"),
            OptionChoice(name="3:2", value="3:2"),
            OptionChoice(name="2:3", value="2:3"),
        ],
    )
    async def image(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "grok-imagine-image-pro",
        aspect_ratio: str = "1:1",
    ):
        """
        Generates an image given a prompt using Grok Imagine.
        """
        await ctx.defer()

        try:
            self.logger.info(f"Generating image with model {model}")
            result = await self.client.image.sample(
                prompt=prompt,
                model=model,
                aspect_ratio=cast(ImageAspectRatio, aspect_ratio),
            )

            image_cost = calculate_image_cost(model)
            daily_cost = self._track_daily_cost_flat(ctx.author.id, image_cost)

            if result.url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(result.url) as resp:
                        if resp.status != 200:
                            raise Exception(f"Failed to download image: HTTP {resp.status}")
                        data = io.BytesIO(await resp.read())

                description = f"**Prompt:** {truncate_text(prompt, 2000)}\n"
                description += f"**Model:** {model}\n"
                description += f"**Aspect Ratio:** {aspect_ratio}\n"

                embed = Embed(
                    title="Image Generation",
                    description=description,
                    color=Colour.dark_teal(),
                )
                file = File(data, "image.png")
                embed.set_image(url="attachment://image.png")
                embeds = [embed]
                append_generation_pricing_embed(embeds, image_cost, daily_cost)
                await ctx.send_followup(embeds=embeds, file=file)
                self.logger.info("Successfully generated and sent image")

            elif result.base64:
                import base64
                image_bytes = base64.b64decode(result.base64)
                data = io.BytesIO(image_bytes)

                description = f"**Prompt:** {truncate_text(prompt, 2000)}\n"
                description += f"**Model:** {model}\n"
                description += f"**Aspect Ratio:** {aspect_ratio}\n"

                embed = Embed(
                    title="Image Generation",
                    description=description,
                    color=Colour.dark_teal(),
                )
                file = File(data, "image.png")
                embed.set_image(url="attachment://image.png")
                embeds = [embed]
                append_generation_pricing_embed(embeds, image_cost, daily_cost)
                await ctx.send_followup(embeds=embeds, file=file)
                self.logger.info("Successfully generated and sent image")

            else:
                raise Exception("No image data returned from the API.")

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(f"Image generation failed: {description}", exc_info=True)
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

    @grok.command(
        name="video",
        description="Generates a video from a prompt.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "aspect_ratio",
        description="Aspect ratio of the video. (default: 16:9)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="16:9 (Landscape)", value="16:9"),
            OptionChoice(name="9:16 (Portrait)", value="9:16"),
            OptionChoice(name="1:1 (Square)", value="1:1"),
            OptionChoice(name="4:3", value="4:3"),
            OptionChoice(name="3:4", value="3:4"),
        ],
    )
    @option(
        "duration",
        description="Duration of the video in seconds. (default: 5)",
        required=False,
        type=int,
    )
    @option(
        "resolution",
        description="Resolution of the video. (default: 720p)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="720p", value="720p"),
            OptionChoice(name="480p", value="480p"),
        ],
    )
    async def video(
        self,
        ctx: ApplicationContext,
        prompt: str,
        aspect_ratio: str = "16:9",
        duration: int = 5,
        resolution: str = "720p",
    ):
        """
        Generates a video from a prompt using Grok Imagine Video.
        """
        await ctx.defer()

        try:
            self.logger.info("Starting video generation with grok-imagine-video")
            result = await self.client.video.generate(
                prompt=prompt,
                model="grok-imagine-video",
                aspect_ratio=cast(VideoAspectRatio, aspect_ratio),
                duration=duration,
                resolution=cast(VideoResolution, resolution),
            )

            if not result.url:
                raise Exception("No video URL returned from the API.")

            async with aiohttp.ClientSession() as session:
                async with session.get(result.url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download video: HTTP {resp.status}")
                    video_bytes = await resp.read()

            video_cost = calculate_video_cost(duration)
            daily_cost = self._track_daily_cost_flat(ctx.author.id, video_cost)

            data = io.BytesIO(video_bytes)
            description = f"**Prompt:** {truncate_text(prompt, 2000)}\n"
            description += f"**Aspect Ratio:** {aspect_ratio}\n"
            description += f"**Duration:** {duration}s\n"
            description += f"**Resolution:** {resolution}\n"

            embed = Embed(
                title="Video Generation",
                description=description,
                color=Colour.dark_teal(),
            )
            embeds = [embed]
            append_generation_pricing_embed(embeds, video_cost, daily_cost)
            await ctx.send_followup(embeds=embeds, file=File(data, "video.mp4"))
            self.logger.info("Successfully generated and sent video")

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(
                f"Video generation failed: {description}", exc_info=True
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

    @grok.command(
        name="tts",
        description="Converts text to speech audio.",
    )
    @option("text", description="Text to convert to speech. (max 15,000 characters)", required=True, type=str)
    @option(
        "voice",
        description="Voice to use for synthesis. (default: Eve)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="Eve (Energetic, upbeat)", value="eve"),
            OptionChoice(name="Ara (Warm, friendly)", value="ara"),
            OptionChoice(name="Rex (Confident, clear)", value="rex"),
            OptionChoice(name="Sal (Smooth, balanced)", value="sal"),
            OptionChoice(name="Leo (Authoritative, strong)", value="leo"),
        ],
    )
    @option(
        "language",
        description="BCP-47 language code, e.g. en, zh, ja, fr, de, es-ES. (default: en)",
        required=False,
        type=str,
    )
    @option(
        "output_format",
        description="Audio output format. (default: mp3)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="MP3", value="mp3"),
            OptionChoice(name="WAV", value="wav"),
        ],
    )
    async def tts(
        self,
        ctx: ApplicationContext,
        text: str,
        voice: str = "eve",
        language: str = "en",
        output_format: str = "mp3",
    ):
        """
        Converts text to speech audio using the xAI TTS API.
        """
        await ctx.defer()

        try:
            if len(text) > TTS_MAX_CHARS:
                await ctx.send_followup(
                    embed=Embed(
                        title="Error",
                        description=f"Text exceeds the {TTS_MAX_CHARS:,} character limit ({len(text):,} characters provided).",
                        color=Colour.red(),
                    )
                )
                return

            self.logger.info(
                f"Generating TTS with voice={voice}, language={language}, format={output_format}"
            )
            audio_bytes = await self._generate_tts(text, voice, language, output_format)

            description = f"**Text:** {truncate_text(text, 2000)}\n"
            description += f"**Voice:** {voice}\n"
            description += f"**Language:** {language}\n"
            description += f"**Format:** {output_format}\n"

            embed = Embed(
                title="Text-to-Speech Generation",
                description=description,
                color=Colour.dark_teal(),
            )
            data = io.BytesIO(audio_bytes)
            await ctx.send_followup(
                embed=embed, file=File(data, f"speech.{output_format}")
            )
            self.logger.info("Successfully generated and sent TTS audio")

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(
                f"TTS generation failed: {description}", exc_info=True
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )
