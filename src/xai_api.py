import asyncio
import io
import logging
import re
from datetime import date, datetime
from typing import Any, Literal, TypedDict, cast

import aiohttp

from xai_sdk import AsyncClient
from xai_sdk.image import ImageAspectRatio
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
    MULTI_AGENT_MODELS,
    PENALTY_SUPPORTED_MODELS,
    REASONING_EFFORT_MODELS,
    TOOL_BUILDERS,
    TOOL_COLLECTIONS_SEARCH,
    TOOL_USAGE_DISPLAY_NAMES,
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
RESPONSES_API_URL = "https://api.x.ai/v1/responses"

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MiB xAI image understanding limit
MAX_FILE_SIZE = 48 * 1024 * 1024  # 48 MB xAI Files API limit

_CITATION_MARKER_RE = re.compile(r"\[\[\d+\]\]\([^)]+\)")


class CitationInfo(TypedDict):
    url: str
    source: Literal["web", "x", "collections"]


class ToolInfo(TypedDict):
    citations: list[CitationInfo]


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
                color=Colour(0x000000),
            )
        )


def _classify_citation_url(url: str) -> Literal["web", "x", "collections"]:
    """Classify a citation URL into its source type."""
    if url.startswith("collections://"):
        return "collections"
    if url.startswith("https://x.com/") or url.startswith("https://twitter.com/"):
        return "x"
    return "web"


def extract_tool_info(response_json: dict[str, Any]) -> ToolInfo:
    """Extract structured citation data from a Responses API JSON response."""
    citations: list[CitationInfo] = []
    seen_urls: set[str] = set()

    for output_item in response_json.get("output", []):
        for content_part in output_item.get("content", []) if isinstance(output_item, dict) else []:
            for annotation in content_part.get("annotations", []) if isinstance(content_part, dict) else []:
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("type") != "url_citation":
                    continue
                url = str(annotation.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                citations.append({"url": url, "source": _classify_citation_url(url)})

    return {"citations": citations}


def append_sources_embed(embeds: list[Embed], citations: list[CitationInfo]) -> None:
    """Append a compact sources embed for tool-backed responses, grouped by type."""
    if not citations or len(embeds) >= 10:
        return

    web: list[CitationInfo] = []
    x: list[CitationInfo] = []
    collections: list[CitationInfo] = []
    for cit in citations:
        if cit["source"] == "x":
            x.append(cit)
        elif cit["source"] == "collections":
            collections.append(cit)
        else:
            web.append(cit)

    parts: list[str] = []

    def _format_link_group(
        heading: str | None, items: list[CitationInfo], limit: int = 8
    ) -> None:
        if not items:
            return
        lines: list[str] = []
        if heading:
            lines.append(f"**{heading}**")
        for index, cit in enumerate(items[:limit], start=1):
            url = cit["url"]
            if url.startswith("http://") or url.startswith("https://"):
                title = truncate_text(
                    url.removeprefix("https://").removeprefix("http://"), 120
                )
                lines.append(f"{index}. [{title}]({url})")
            else:
                lines.append(f"{index}. `{truncate_text(url, 300)}`")
        parts.append("\n".join(lines))

    # If only one type exists, skip the heading for a cleaner look
    has_multiple_types = sum(bool(g) for g in (web, x, collections)) > 1

    _format_link_group("Web" if has_multiple_types else None, web)
    _format_link_group("X Posts" if has_multiple_types else None, x)
    _format_link_group("Collections" if has_multiple_types else None, collections)

    description = "\n\n".join(parts)
    if len(description) > 4000:
        description = truncate_text(description, 3990)

    embeds.append(
        Embed(
            title="Sources",
            description=description,
            color=Colour(0x000000),
        )
    )


def append_pricing_embed(
    embeds: list[Embed],
    model: str,
    input_tokens: int,
    output_tokens: int,
    daily_cost: float,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    image_tokens: int = 0,
    tool_usage: dict[str, int] | None = None,
) -> None:
    """Append a compact pricing embed showing cost and token usage."""
    cost = calculate_cost(model, input_tokens, output_tokens, reasoning_tokens)
    in_qualifiers = []
    if cached_tokens > 0:
        in_qualifiers.append(f"{cached_tokens:,} cached")
    if image_tokens > 0:
        in_qualifiers.append(f"{image_tokens:,} image")
    token_info = f"{input_tokens:,} tokens in"
    if in_qualifiers:
        token_info += f" ({', '.join(in_qualifiers)})"
    token_info += f" / {output_tokens:,} tokens out"
    if reasoning_tokens > 0:
        token_info += f" ({reasoning_tokens:,} reasoning)"
    description = f"${cost:.4f} · {token_info} · daily ${daily_cost:.2f}"
    if tool_usage:
        tool_parts = []
        for key, count in tool_usage.items():
            name = TOOL_USAGE_DISPLAY_NAMES.get(
                key, key.replace("SERVER_SIDE_TOOL_", "").replace("_", " ").title()
            )
            tool_parts.append(f"{name} \u00d7{count}")
        description += "\n" + " \u00b7 ".join(tool_parts)
    embeds.append(Embed(description=description, color=Colour(0x000000)))


def append_generation_pricing_embed(
    embeds: list[Embed],
    cost: float,
    daily_cost: float,
) -> None:
    """Append a compact pricing embed for image/video generation."""
    description = f"${cost:.4f} · daily ${daily_cost:.2f}"
    embeds.append(Embed(description=description, color=Colour(0x000000)))


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
        # Last message with a ButtonView attached, keyed by user — used to strip old buttons
        self.last_view_messages = {}
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
        self,
        user_id: int,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> float:
        """Add this request's cost to the user's daily total and return the new daily total."""
        cost = calculate_cost(model, input_tokens, output_tokens, reasoning_tokens)
        key = (user_id, date.today().isoformat())
        self.daily_costs[key] = self.daily_costs.get(key, 0.0) + cost
        return self.daily_costs[key]

    def _track_daily_cost_flat(self, user_id: int, cost: float) -> float:
        """Add a flat cost to the user's daily total and return the new daily total."""
        key = (user_id, date.today().isoformat())
        self.daily_costs[key] = self.daily_costs.get(key, 0.0) + cost
        return self.daily_costs[key]

    async def _strip_previous_view(self, user) -> None:
        """Edit the last message that had buttons to remove its view."""
        prev = self.last_view_messages.pop(user, None)
        if prev is not None:
            try:
                await prev.edit(view=None)
            except Exception:
                pass  # Message may have been deleted or is no longer editable

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

    async def _call_responses_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the xAI Responses API and return the parsed JSON response."""
        session = await self._get_http_session()
        async with session.post(
            RESPONSES_API_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                raise Exception(
                    f"Responses API error (HTTP {resp.status}): {error_body}"
                )
            return await resp.json()

    def _build_responses_payload(
        self,
        model: str,
        input_messages: list[dict[str, Any]],
        *,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        reasoning_effort: str | None = None,
        agent_count: int | None = None,
    ) -> dict[str, Any]:
        """Build a JSON payload for the Responses API."""
        payload: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "store": True,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if tools:
            payload["tools"] = tools
        if tools or model in MULTI_AGENT_MODELS:
            payload["include"] = ["reasoning.encrypted_content"]
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if agent_count is not None:
            payload["agent_count"] = agent_count
        return payload

    @staticmethod
    def _build_user_message(content_parts: list[Any]) -> dict[str, Any]:
        """Build a user message from a list of content parts."""
        if len(content_parts) == 1 and isinstance(content_parts[0], str):
            return {"role": "user", "content": content_parts[0]}
        content = []
        for part in content_parts:
            if isinstance(part, str):
                content.append({"type": "input_text", "text": part})
            elif isinstance(part, dict):
                content.append(part)
        return {"role": "user", "content": content}

    @staticmethod
    def _extract_response_text(response_json: dict[str, Any]) -> tuple[str, str]:
        """Extract response text and reasoning text from a Responses API response."""
        response_text = ""
        reasoning_text = ""
        for output_item in response_json.get("output", []):
            if not isinstance(output_item, dict):
                continue
            if output_item.get("type") == "reasoning":
                for part in output_item.get("summary", []):
                    if isinstance(part, dict) and part.get("type") == "summary_text":
                        reasoning_text += part.get("text", "")
            elif output_item.get("role") == "assistant":
                for content_part in output_item.get("content", []):
                    if isinstance(content_part, dict) and content_part.get("type") == "output_text":
                        response_text += content_part.get("text", "")
        response_text = _CITATION_MARKER_RE.sub("", response_text).strip()
        return response_text or "No response.", reasoning_text

    @staticmethod
    def _extract_usage(response_json: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from a Responses API response.

        The Responses API uses ``input_tokens`` / ``output_tokens`` and
        ``input_tokens_details`` / ``output_tokens_details``.  We also
        accept the Chat Completions names as a fallback.
        """
        usage = response_json.get("usage", {})
        input_details = (
            usage.get("input_tokens_details")
            or usage.get("prompt_tokens_details")
            or {}
        )
        output_details = (
            usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )
        return {
            "input_tokens": (
                usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            ),
            "output_tokens": (
                usage.get("output_tokens") or usage.get("completion_tokens") or 0
            ),
            "reasoning_tokens": output_details.get("reasoning_tokens", 0) or 0,
            "cached_tokens": input_details.get("cached_tokens", 0) or 0,
            "image_tokens": input_details.get("image_tokens", 0) or 0,
        }

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
            starter = conversation.params.conversation_starter
            if starter is not None:
                self.last_view_messages.pop(starter, None)
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
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Build Responses API tool dicts for the selected tool names."""
        tools: list[dict[str, Any]] = []

        for tool_name in selected_tool_names:
            if tool_name == TOOL_COLLECTIONS_SEARCH:
                if not XAI_COLLECTION_IDS:
                    return (
                        [],
                        "Collections search requires XAI_COLLECTION_IDS to be set in your .env.",
                    )
                tools.append({
                    "type": "file_search",
                    "vector_store_ids": XAI_COLLECTION_IDS.copy(),
                })
                continue

            if tool_name == TOOL_X_SEARCH and x_search_kwargs:
                # Convert datetime objects to ISO strings for REST
                kw = dict(x_search_kwargs)
                for date_key in ("from_date", "to_date"):
                    if isinstance(kw.get(date_key), datetime):
                        kw[date_key] = kw[date_key].isoformat()
                tools.append({"type": "x_search", **kw})
                continue

            if tool_name == TOOL_WEB_SEARCH and web_search_kwargs:
                tools.append({"type": "web_search", **web_search_kwargs})
                continue

            tool_builder = TOOL_BUILDERS.get(tool_name)
            if tool_builder is None:
                continue
            tools.append(tool_builder())

        return tools, None

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
            content_parts: list[Any] = []
            if message.content:
                content_parts.append(message.content)
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type in SUPPORTED_IMAGE_TYPES:
                        if attachment.size > MAX_IMAGE_SIZE:
                            self.logger.warning(
                                "Image %s exceeds 20 MiB limit (%s bytes), skipping",
                                attachment.filename,
                                attachment.size,
                            )
                            continue
                        content_parts.append({"type": "input_image", "image_url": attachment.url, "detail": "high"})
                    else:
                        file_id = await self._upload_file_attachment(attachment)
                        if file_id:
                            conversation.file_ids.append(file_id)
                            content_parts.append({"type": "input_file", "file_id": file_id})

            if not content_parts:
                return

            input_messages = [self._build_user_message(content_parts)]
            payload = self._build_responses_payload(
                model=params.model,
                input_messages=input_messages,
                previous_response_id=conversation.previous_response_id,
                tools=params.tools or None,
                max_output_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                reasoning_effort=params.reasoning_effort,
                agent_count=params.agent_count,
            )
            response_json = await self._call_responses_api(payload)
            response_text, reasoning_text = self._extract_response_text(response_json)
            tool_info = extract_tool_info(response_json)

            usage = self._extract_usage(response_json)
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            reasoning_tokens = usage["reasoning_tokens"]
            cached_tokens = usage["cached_tokens"]
            image_tokens = usage["image_tokens"]
            tool_usage = response_json.get("server_side_tool_usage", {})

            # Stop typing as soon as we have the response
            if typing_task:
                typing_task.cancel()
                typing_task = None

            # Update conversation state
            response_id = response_json.get("id")
            if response_id:
                conversation.response_id_history.append(response_id)
                conversation.previous_response_id = response_id

            append_reasoning_embeds(embeds, reasoning_text)
            append_response_embeds(embeds, response_text)

            append_sources_embed(embeds, tool_info["citations"])
            daily_cost = self._track_daily_cost(
                message.author.id, params.model, input_tokens, output_tokens, reasoning_tokens
            )
            if SHOW_COST_EMBEDS:
                append_pricing_embed(
                    embeds,
                    params.model,
                    input_tokens,
                    output_tokens,
                    daily_cost,
                    reasoning_tokens,
                    cached_tokens,
                    image_tokens,
                    tool_usage,
                )

            view = self.views.get(message.author)
            main_conversation_id = params.conversation_id

            if main_conversation_id is None:
                self.logger.error("Conversation ID is None, cannot track message")
                return

            # Strip buttons from previous turn's message
            await self._strip_previous_view(message.author)

            if embeds:
                try:
                    reply_message = await message.reply(embeds=embeds, view=view)
                    self.message_to_conversation_id[reply_message.id] = (
                        main_conversation_id
                    )
                    self.last_view_messages[message.author] = reply_message
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
                    self.last_view_messages[message.author] = reply_message

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
        "agent_count",
        description="Number of agents for multi-agent model. 4=quick, 16=deep research. (default: not set)",
        required=False,
        type=int,
        choices=[
            OptionChoice(name="4 Agents (Quick)", value=4),
            OptionChoice(name="16 Agents (Deep Research)", value=16),
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
        agent_count: int | None = None,
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
                        description=f"`agent_count` is only supported by multi-agent models ({', '.join(f'`{m}`' for m in sorted(MULTI_AGENT_MODELS))}).",
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
            initial_messages: list[dict[str, Any]] = []
            if system_prompt:
                initial_messages.append({"role": "system", "content": system_prompt})

            # Build user message content parts
            uploaded_file_ids: list[str] = []
            content_parts: list[Any] = [prompt]
            if attachment:
                if attachment.content_type in SUPPORTED_IMAGE_TYPES:
                    if attachment.size > MAX_IMAGE_SIZE:
                        await ctx.followup.send(
                            f"Image `{attachment.filename}` exceeds the 20 MiB limit.",
                            ephemeral=True,
                        )
                        return
                    content_parts.append({"type": "input_image", "image_url": attachment.url, "detail": "high"})
                else:
                    file_id = await self._upload_file_attachment(attachment)
                    if file_id:
                        uploaded_file_ids.append(file_id)
                        content_parts.append({"type": "input_file", "file_id": file_id})
            initial_messages.append(self._build_user_message(content_parts))

            payload = self._build_responses_payload(
                model=model,
                input_messages=initial_messages,
                tools=tools or None,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                agent_count=agent_count if is_multi_agent else None,
            )
            response_json = await self._call_responses_api(payload)
            response_text, reasoning_text = self._extract_response_text(response_json)
            tool_info = extract_tool_info(response_json)

            usage = self._extract_usage(response_json)
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            reasoning_tokens = usage["reasoning_tokens"]
            cached_tokens = usage["cached_tokens"]
            image_tokens = usage["image_tokens"]
            tool_usage = response_json.get("server_side_tool_usage", {})

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
            if agent_count is not None:
                description += f"**Agent Count:** {agent_count}\n"
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
                ctx.author.id, model, input_tokens, output_tokens, reasoning_tokens
            )
            if SHOW_COST_EMBEDS:
                append_pricing_embed(
                    embeds,
                    model,
                    input_tokens,
                    output_tokens,
                    daily_cost,
                    reasoning_tokens,
                    cached_tokens,
                    image_tokens,
                    tool_usage,
                )

            if len(embeds) == 1:
                await ctx.send_followup("No response generated.")
                return

            # Create the view with buttons
            main_conversation_id = ctx.interaction.id

            # Strip buttons from any prior conversation's last message
            await self._strip_previous_view(ctx.author)

            view = ButtonView(
                cog=self,
                conversation_starter=ctx.author,
                conversation_id=main_conversation_id,
                initial_tools=tools,
            )
            self.views[ctx.author] = view

            msg = await ctx.send_followup(embeds=embeds, view=view)
            self.message_to_conversation_id[msg.id] = main_conversation_id
            self.last_view_messages[ctx.author] = msg

            # Store the conversation
            response_id = response_json.get("id")
            params = ChatCompletionParameters(
                model=model,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                agent_count=agent_count,
                tools=tools,
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
                if SHOW_COST_EMBEDS:
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
                if SHOW_COST_EMBEDS:
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
            if SHOW_COST_EMBEDS:
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
