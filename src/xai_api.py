import asyncio
import io
import logging
from typing import Any, cast

import aiohttp

from xai_sdk import AsyncClient
from xai_sdk.chat import image, system, user
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
from config.auth import GUILD_IDS, XAI_API_KEY
from util import (
    ChatCompletionParameters,
    Conversation,
    REASONING_MODELS,
    chunk_text,
    format_xai_error,
    truncate_text,
)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


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


class xAIAPI(commands.Cog):
    xai = SlashCommandGroup("xai", "xAI Grok commands", guild_ids=GUILD_IDS)

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

        # Dictionary to store conversation state for each converse interaction
        self.conversations: dict[int, Conversation] = {}
        # Dictionary to map any message ID to the main conversation ID for tracking
        self.message_to_conversation_id: dict[int, int] = {}
        # Dictionary to store UI views for each conversation
        self.views = {}
        self._http_session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session and not self._http_session.closed:
            return self._http_session
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession()
            return self._http_session

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

            if content_parts:
                chat.append(user(*content_parts))

            response = await chat.sample()
            response_text = response.content or "No response."
            reasoning_text = response.reasoning_content or ""

            # Stop typing as soon as we have the response
            if typing_task:
                typing_task.cancel()
                typing_task = None

            # Add assistant response to chat history
            chat.append(response)

            append_reasoning_embeds(embeds, reasoning_text)
            append_response_embeds(embeds, response_text)

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

    @xai.command(
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

    @xai.command(
        name="converse",
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
        description="Choose from the following Grok models. (default: Grok 4.1 Fast Reasoning)",
        required=False,
        choices=[
            OptionChoice(name="Grok 4.1 Fast Reasoning", value="grok-4-1-fast-reasoning"),
            OptionChoice(name="Grok 4.1 Fast Non-Reasoning", value="grok-4-1-fast-non-reasoning"),
            OptionChoice(name="Grok Code Fast 1", value="grok-code-fast-1"),
            OptionChoice(name="Grok 4 Fast Reasoning", value="grok-4-fast-reasoning"),
            OptionChoice(name="Grok 4 Fast Non-Reasoning", value="grok-4-fast-non-reasoning"),
            OptionChoice(name="Grok 4 (0709)", value="grok-4-0709"),
            OptionChoice(name="Grok 3 Mini", value="grok-3-mini"),
            OptionChoice(name="Grok 3", value="grok-3"),
            OptionChoice(name="Grok 2 Vision (1212)", value="grok-2-vision-1212"),
        ],
        type=str,
    )
    @option(
        "attachment",
        description="Attach an image (JPEG, PNG, GIF, WEBP).",
        required=False,
        type=Attachment,
    )
    @option(
        "max_tokens",
        description="Maximum tokens in the response. (default: 16384)",
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
        description="(Advanced) Controls how much the model should repeat itself. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "presence_penalty",
        description="(Advanced) Controls how much the model should talk about new topics. (default: not set)",
        required=False,
        type=float,
    )
    async def converse(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "grok-4-1-fast-reasoning",
        system_prompt: str | None = None,
        attachment: Attachment | None = None,
        max_tokens: int = 16384,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
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

            # Build initial messages
            initial_messages = []
            if system_prompt:
                initial_messages.append(system(system_prompt))

            # Build user message content parts
            content_parts: list[Any] = [prompt]
            if attachment and attachment.content_type in SUPPORTED_IMAGE_TYPES:
                content_parts.append(image(attachment.url))
            initial_messages.append(user(*content_parts))

            # Build create kwargs
            create_kwargs = {
                "model": model,
                "messages": initial_messages,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            if top_p is not None:
                create_kwargs["top_p"] = top_p
            if frequency_penalty is not None:
                create_kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                create_kwargs["presence_penalty"] = presence_penalty
            if model in REASONING_MODELS:
                create_kwargs["reasoning_effort"] = "high"

            chat = self.client.chat.create(**create_kwargs)
            response = await chat.sample()
            response_text = response.content or "No response."
            reasoning_text = response.reasoning_content or ""

            # Add assistant response to chat history
            chat.append(response)

            # Build description embed
            truncated_prompt = truncate_text(prompt, 2000)
            description = f"**Prompt:** {truncated_prompt}\n"
            description += f"**Model:** {model}\n"
            if system_prompt:
                description += f"**System:** {truncate_text(system_prompt, 500)}\n"
            description += f"**Max Tokens:** {max_tokens}\n"
            if temperature is not None:
                description += f"**Temperature:** {temperature}\n"
            if top_p is not None:
                description += f"**Top P:** {top_p}\n"
            if frequency_penalty is not None:
                description += f"**Frequency Penalty:** {frequency_penalty}\n"
            if presence_penalty is not None:
                description += f"**Presence Penalty:** {presence_penalty}\n"

            embeds = [
                Embed(
                    title="Conversation Started",
                    description=description,
                    color=Colour.green(),
                )
            ]
            append_reasoning_embeds(embeds, reasoning_text)
            append_response_embeds(embeds, response_text)

            if len(embeds) == 1:
                await ctx.send_followup("No response generated.")
                return

            # Create the view with buttons
            main_conversation_id = ctx.interaction.id
            view = ButtonView(
                cog=self,
                conversation_starter=ctx.author,
                conversation_id=main_conversation_id,
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
                conversation_starter=ctx.author,
                channel_id=ctx.channel.id,
                conversation_id=main_conversation_id,
            )
            conversation = Conversation(params=params, chat=chat)
            self.conversations[main_conversation_id] = conversation

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(
                f"Error in converse: {description}",
                exc_info=True,
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

        finally:
            if typing_task:
                typing_task.cancel()

    @xai.command(
        name="image",
        description="Generates an image from a prompt.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "model",
        description="Choose from the following image generation models. (default: Grok Imagine Image)",
        required=False,
        type=str,
        choices=[
            OptionChoice(name="Grok Imagine Image Pro", value="grok-imagine-image-pro"),
            OptionChoice(name="Grok Imagine Image", value="grok-imagine-image"),
            OptionChoice(name="Grok 2 Image (1212)", value="grok-2-image-1212"),
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
    async def generate_image(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "grok-imagine-image",
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
                await ctx.send_followup(embed=embed, file=file)
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
                await ctx.send_followup(embed=embed, file=file)
                self.logger.info("Successfully generated and sent image")

            else:
                raise Exception("No image data returned from the API.")

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(f"Image generation failed: {description}", exc_info=True)
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )

    @xai.command(
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
    async def generate_video(
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
            await ctx.send_followup(embed=embed, file=File(data, "video.mp4"))
            self.logger.info("Successfully generated and sent video")

        except Exception as e:
            description = format_xai_error(e)
            self.logger.error(
                f"Video generation failed: {description}", exc_info=True
            )
            await ctx.send_followup(
                embed=Embed(title="Error", description=description, color=Colour.red())
            )
