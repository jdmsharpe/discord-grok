import asyncio
import logging
from typing import Any

import aiohttp
from discord import ApplicationContext, Attachment, Bot, Member, Message, User
from discord.commands import OptionChoice, SlashCommandGroup, option
from discord.ext import commands

from ...config.auth import GUILD_IDS, SHOW_COST_EMBEDS
from .attachments import (
    MAX_FILE_SIZE,
    MAX_IMAGE_SIZE,
    build_user_message,
    fetch_attachment_bytes,
    unsupported_image_type_error,
)
from .chat import (
    handle_check_permissions,
    handle_on_message,
    run_chat_command,
)
from .chat import (
    handle_new_message_in_conversation as run_followup_message,
)
from .chat import (
    keep_typing as keep_typing_loop,
)
from .client import (
    MAX_API_ATTEMPTS,
    RESPONSES_API_URL,
    RETRYABLE_STATUS_CODES,
    TTS_API_URL,
    TTS_MAX_CHARS,
    build_responses_payload,
    build_xai_headers,
    call_responses_api,
    call_tts_api,
    cleanup_conversation_files,
    close_http_session,
    compute_retry_delay,
    describe_api_request,
    generate_tts,
    get_client,
    get_http_session,
    parse_retry_after,
    post_with_retries,
    upload_file_attachment,
)
from .command_options import (
    DEFAULT_CHAT_MODEL_ENTRY,
    DEFAULT_CHAT_MODEL_ID,
    iter_slash_command_models,
)
from .embeds import (
    GROK_BLACK,
    append_generation_pricing_embed,
    append_pricing_embed,
    append_reasoning_embeds,
    append_response_embeds,
    append_sources_embed,
)
from .image import run_image_command
from .models import ChatCompletionParameters, CitationInfo, Conversation, ToolInfo
from .responses import extract_response_text, extract_tool_info, extract_usage
from .speech import run_tts_command
from .state import (
    end_conversation as end_conversation_state,
)
from .state import (
    log_chat_cost,
    resolve_tools_for_view,
    strip_previous_view,
    track_daily_cost,
)
from .tooling import resolve_selected_tools
from .video import run_video_command
from .views import ButtonView

__all__ = [
    "ChatCompletionParameters",
    "CitationInfo",
    "Conversation",
    "GrokCog",
    "GROK_BLACK",
    "MAX_API_ATTEMPTS",
    "MAX_FILE_SIZE",
    "MAX_IMAGE_SIZE",
    "RESPONSES_API_URL",
    "RETRYABLE_STATUS_CODES",
    "TTS_API_URL",
    "TTS_MAX_CHARS",
    "ToolInfo",
    "append_generation_pricing_embed",
    "append_pricing_embed",
    "append_reasoning_embeds",
    "append_response_embeds",
    "append_sources_embed",
    "extract_response_text",
    "extract_tool_info",
    "extract_usage",
]

CHAT_MODEL_CHOICES = [
    OptionChoice(name=entry.display_name, value=entry.model_id)
    for entry in iter_slash_command_models()
]

REASONING_EFFORT_CHOICES = [
    OptionChoice(name="Low", value="low"),
    OptionChoice(name="High", value="high"),
]

AGENT_COUNT_CHOICES = [
    OptionChoice(name="4 Agents (Quick)", value=4),
    OptionChoice(name="16 Agents (Deep Research)", value=16),
]


class GrokCog(commands.Cog):
    grok = SlashCommandGroup("grok", "xAI Grok commands", guild_ids=GUILD_IDS)

    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.client: Any | None = None
        self.logger = logging.getLogger(__name__)
        self.show_cost_embeds = SHOW_COST_EMBEDS
        self.conversations: dict[int, Conversation] = {}
        self.views: dict[Member | User, ButtonView] = {}
        self.last_view_messages: dict[Member | User, Message] = {}
        self.daily_costs: dict[tuple[int, str], float] = {}
        self._http_session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    def _get_client(self) -> Any:
        return get_client(self)

    async def _get_http_session(self) -> aiohttp.ClientSession:
        return await get_http_session(self)

    @staticmethod
    def _build_xai_headers(*, grok_conv_id: str | None = None) -> dict[str, str]:
        return build_xai_headers(grok_conv_id=grok_conv_id)

    @staticmethod
    def _describe_api_request(url: str) -> str:
        return describe_api_request(url)

    @staticmethod
    def _parse_retry_after(retry_after: str | None) -> float | None:
        return parse_retry_after(retry_after)

    def _compute_retry_delay(self, attempt: int, *, retry_after: str | None = None) -> float:
        return compute_retry_delay(attempt, retry_after=retry_after)

    async def _post_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        json_payload: dict[str, Any],
    ) -> bytes:
        return await post_with_retries(self, url, headers, json_payload)

    def _track_daily_cost(self, user_id: int, cost: float) -> float:
        return track_daily_cost(self, user_id, cost)

    def _log_chat_cost(
        self,
        user_id: int,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        image_tokens: int,
        tool_usage: dict[str, int],
        request_cost: float,
        daily_cost: float,
    ) -> None:
        log_chat_cost(
            self,
            user_id,
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

    async def _strip_previous_view(self, user: Member | User) -> None:
        await strip_previous_view(self, user)

    async def _generate_tts(
        self,
        text: str,
        voice_id: str,
        language: str,
        codec: str,
        sample_rate: int | None = None,
        bit_rate: int | None = None,
    ) -> bytes:
        return await generate_tts(self, text, voice_id, language, codec, sample_rate, bit_rate)

    async def _call_tts_api(self, payload: dict[str, Any]) -> bytes:
        return await call_tts_api(self, payload)

    async def _call_responses_api(
        self,
        payload: dict[str, Any],
        *,
        grok_conv_id: str | None = None,
    ) -> dict[str, Any]:
        return await call_responses_api(self, payload, grok_conv_id=grok_conv_id)

    @staticmethod
    def _build_responses_payload(
        model: str,
        input_messages: list[dict[str, Any]],
        *,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        reasoning_effort: str | None = None,
        agent_count: int | None = None,
        include_encrypted_reasoning: bool = False,
    ) -> dict[str, Any]:
        return build_responses_payload(
            model,
            input_messages,
            previous_response_id=previous_response_id,
            prompt_cache_key=prompt_cache_key,
            tools=tools,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            agent_count=agent_count,
            include_encrypted_reasoning=include_encrypted_reasoning,
        )

    @staticmethod
    def _build_user_message(content_parts: list[Any]) -> dict[str, Any]:
        return build_user_message(content_parts)

    @staticmethod
    def _extract_response_text(response_json: dict[str, Any]) -> tuple[str, str]:
        return extract_response_text(response_json)

    @staticmethod
    def _extract_tool_info(response_json: dict[str, Any]) -> ToolInfo:
        return extract_tool_info(response_json)

    @staticmethod
    def _extract_usage(response_json: dict[str, Any]) -> dict[str, int]:
        return extract_usage(response_json)

    async def _fetch_attachment_bytes(self, attachment: Attachment) -> bytes | None:
        return await fetch_attachment_bytes(self, attachment)

    @staticmethod
    def _unsupported_image_type_error(attachment: Attachment) -> str | None:
        return unsupported_image_type_error(attachment)

    async def _upload_file_attachment(self, attachment: Attachment) -> str | None:
        return await upload_file_attachment(
            self,
            attachment,
            fetch_bytes=self._fetch_attachment_bytes,
        )

    async def _cleanup_conversation_files(self, conversation: Conversation) -> None:
        await cleanup_conversation_files(self, conversation)

    def _resolve_tools_for_view(
        self,
        selected_values: list[str],
        conversation: Conversation,
    ) -> tuple[set[str], str | None]:
        return resolve_tools_for_view(self, selected_values, conversation)

    async def end_conversation(self, conversation_id: int) -> None:
        await end_conversation_state(self, conversation_id)

    async def _close_http_session(self) -> None:
        await close_http_session(self)

    def cog_unload(self) -> None:
        loop = getattr(self.bot, "loop", None)
        session = self._http_session
        if session and not session.closed:
            if loop and loop.is_running():
                loop.create_task(self._close_http_session())
            else:
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(self._close_http_session())
                finally:
                    new_loop.close()
        self._http_session = None

    def resolve_selected_tools(
        self,
        selected_tool_names: list[str],
        x_search_kwargs: dict[str, Any] | None = None,
        web_search_kwargs: dict[str, Any] | None = None,
        mcp_servers: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        return resolve_selected_tools(
            selected_tool_names,
            x_search_kwargs=x_search_kwargs,
            web_search_kwargs=web_search_kwargs,
            mcp_servers=mcp_servers,
        )

    async def handle_new_message_in_conversation(
        self,
        message: Message,
        conversation: Conversation,
    ) -> None:
        await run_followup_message(self, message, conversation)

    async def keep_typing(self, channel: Any) -> None:
        await keep_typing_loop(self, channel)

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        bot_user = self.bot.user
        self.logger.info(
            "Logged in as %s (ID: %s)",
            bot_user,
            bot_user.id if bot_user else "unknown",
        )
        self.logger.info("Attempting to sync commands for guilds: %s", GUILD_IDS)
        try:
            await self.bot.sync_commands()
            self.logger.info("Commands synchronized successfully.")
        except Exception as error:
            self.logger.error("Error during command synchronization: %s", error, exc_info=True)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        await handle_on_message(self, message)

    @commands.Cog.listener()
    async def on_error(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error("Error in event %s: %s %s", event, args, kwargs, exc_info=True)

    @grok.command(
        name="check_permissions",
        description="Check if bot has necessary permissions in this channel",
    )
    async def check_permissions(self, ctx: ApplicationContext) -> None:
        await handle_check_permissions(ctx)

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
        description=(
            "Choose from the following Grok models. "
            f"(default: {DEFAULT_CHAT_MODEL_ENTRY.display_name})"
        ),
        required=False,
        choices=CHAT_MODEL_CHOICES,
        type=str,
    )
    @option(
        "attachment",
        description="Attach an image or document. (default: not set)",
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
        choices=REASONING_EFFORT_CHOICES,
    )
    @option(
        "agent_count",
        description="Number of agents for multi-agent model. 4=quick, 16=deep research. (default: not set)",
        required=False,
        type=int,
        choices=AGENT_COUNT_CHOICES,
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
        "mcp",
        description="Comma-separated MCP preset names to enable. (default: not set)",
        required=False,
        type=str,
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
        "web_search_images",
        description="Allow web search to analyze images on web pages. (default: false)",
        required=False,
        type=bool,
    )
    async def chat(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = DEFAULT_CHAT_MODEL_ID,
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
        await run_chat_command(
            self,
            ctx=ctx,
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            attachment=attachment,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            agent_count=agent_count,
            web_search=web_search,
            x_search=x_search,
            code_execution=code_execution,
            collections_search=collections_search,
            mcp=mcp,
            x_search_images=x_search_images,
            x_search_videos=x_search_videos,
            x_search_date_range=x_search_date_range,
            web_search_images=web_search_images,
        )

    @grok.command(
        name="image",
        description="Generates or edits an image from a prompt.",
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
            OptionChoice(name="2:1", value="2:1"),
            OptionChoice(name="1:2", value="1:2"),
            OptionChoice(name="20:9 (Ultrawide)", value="20:9"),
            OptionChoice(name="9:20", value="9:20"),
            OptionChoice(name="19.5:9 (Mobile)", value="19.5:9"),
            OptionChoice(name="9:19.5", value="9:19.5"),
        ],
    )
    @option(
        "resolution",
        description="Image resolution. (default: 1k)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="1k", value="1k"),
            OptionChoice(name="2k", value="2k"),
        ],
    )
    @option(
        "count",
        description="Number of images to generate (1-10). Not supported for editing. (default: 1)",
        required=False,
        type=int,
        min_value=1,
        max_value=10,
    )
    @option(
        "attachment",
        description="Image to edit or remix. (default: not set)",
        required=False,
        type=Attachment,
    )
    async def image(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "grok-imagine-image-pro",
        aspect_ratio: str = "1:1",
        resolution: str | None = None,
        count: int = 1,
        attachment: Attachment | None = None,
    ) -> None:
        await run_image_command(
            self,
            ctx=ctx,
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            count=count,
            attachment=attachment,
        )

    @grok.command(
        name="video",
        description="Generates a video from a prompt or an image.",
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
            OptionChoice(name="3:2", value="3:2"),
            OptionChoice(name="2:3", value="2:3"),
        ],
    )
    @option(
        "duration",
        description="Duration of the video in seconds (1-15). (default: 5 seconds)",
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
    @option(
        "attachment",
        description="Image to use as the first frame. (default: not set)",
        required=False,
        type=Attachment,
    )
    async def video(
        self,
        ctx: ApplicationContext,
        prompt: str,
        aspect_ratio: str = "16:9",
        duration: int = 5,
        resolution: str = "720p",
        attachment: Attachment | None = None,
    ) -> None:
        await run_video_command(
            self,
            ctx=ctx,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            duration=duration,
            resolution=resolution,
            attachment=attachment,
        )

    @grok.command(
        name="tts",
        description="Converts text to speech audio.",
    )
    @option(
        "text",
        description="Text to convert to speech. (max 15,000 characters)",
        required=True,
        type=str,
    )
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
        description="BCP-47 language code or 'auto' for detection. e.g. en, zh, ja, fr, de. (default: auto)",
        required=False,
        type=str,
    )
    @option(
        "output_format",
        description="Audio codec. (default: mp3)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="MP3", value="mp3"),
            OptionChoice(name="WAV (lossless)", value="wav"),
            OptionChoice(name="PCM (raw)", value="pcm"),
            OptionChoice(name="μ-law (telephony)", value="mulaw"),
            OptionChoice(name="A-law (telephony)", value="alaw"),
        ],
    )
    @option(
        "sample_rate",
        description="Audio sample rate in Hz. (default: 24000)",
        required=False,
        type=int,
        choices=[
            OptionChoice(name="8,000 Hz (narrowband)", value=8000),
            OptionChoice(name="16,000 Hz (wideband)", value=16000),
            OptionChoice(name="22,050 Hz (standard)", value=22050),
            OptionChoice(name="24,000 Hz (high quality)", value=24000),
            OptionChoice(name="44,100 Hz (CD quality)", value=44100),
            OptionChoice(name="48,000 Hz (studio)", value=48000),
        ],
    )
    @option(
        "bit_rate",
        description="MP3 bit rate in bps. Only applies to MP3 codec. (default: 128000)",
        required=False,
        type=int,
        choices=[
            OptionChoice(name="32 kbps (low)", value=32000),
            OptionChoice(name="64 kbps (medium)", value=64000),
            OptionChoice(name="96 kbps (standard)", value=96000),
            OptionChoice(name="128 kbps (high)", value=128000),
            OptionChoice(name="192 kbps (maximum)", value=192000),
        ],
    )
    async def tts(
        self,
        ctx: ApplicationContext,
        text: str,
        voice: str = "eve",
        language: str = "auto",
        output_format: str = "mp3",
        sample_rate: int | None = None,
        bit_rate: int | None = None,
    ) -> None:
        await run_tts_command(
            self,
            ctx=ctx,
            text=text,
            voice=voice,
            language=language,
            output_format=output_format,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
        )


def _refresh_group_command_options(group: SlashCommandGroup) -> None:
    """Work around Pycord parsing grouped cog commands before group attachment."""
    for command in group.subcommands:
        command.attached_to_group = True
        if isinstance(command, SlashCommandGroup):
            _refresh_group_command_options(command)
            continue
        command._validate_parameters()


_refresh_group_command_options(GrokCog.grok)
