from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discord import Colour, Embed


class TestAppendPricingEmbed:
    """Tests for the append_pricing_embed helper."""

    def test_append_pricing_embed(self):
        from src.xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, "grok-3", 1000, 500, 1.50)
        assert len(embeds) == 1
        desc = embeds[0].description
        assert "grok-3" in desc
        assert "1,000 tokens in" in desc
        assert "500 tokens out" in desc
        assert "daily $1.50" in desc
        assert embeds[0].colour == Colour.dark_teal()

    def test_append_pricing_embed_with_reasoning_tokens(self):
        from src.xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, "grok-3-mini", 1000, 500, 1.50, reasoning_tokens=200)
        assert len(embeds) == 1
        assert "200 reasoning" in embeds[0].description

    def test_append_pricing_embed_hides_zero_reasoning_tokens(self):
        from src.xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, "grok-3", 1000, 500, 1.50, reasoning_tokens=0)
        assert "reasoning" not in embeds[0].description

    def test_append_generation_pricing_embed(self):
        from src.xai_api import append_generation_pricing_embed

        embeds: list[Embed] = []
        append_generation_pricing_embed(embeds, 0.07, 2.50)
        assert len(embeds) == 1
        assert "$0.0700" in embeds[0].description
        assert "daily $2.50" in embeds[0].description


class TestTrackDailyCost:
    """Tests for the _track_daily_cost methods."""

    @pytest.fixture
    def cog(self, mock_bot):
        with patch("xai_sdk.AsyncClient"):
            from src.xai_api import xAIAPI

            cog = xAIAPI(bot=mock_bot)
            return cog

    def test_track_daily_cost_accumulates(self, cog):
        # grok-3: $3/M in, $15/M out
        daily = cog._track_daily_cost(1, "grok-3", 1_000_000, 0)
        assert daily == pytest.approx(3.00)
        daily = cog._track_daily_cost(1, "grok-3", 0, 1_000_000)
        assert daily == pytest.approx(18.00)

    def test_track_daily_cost_flat(self, cog):
        daily = cog._track_daily_cost_flat(1, 0.07)
        assert daily == pytest.approx(0.07)
        daily = cog._track_daily_cost_flat(1, 0.02)
        assert daily == pytest.approx(0.09)


class TestExtractToolInfo:
    """Tests for extract_tool_info helper."""

    def test_extract_tool_info_deduplicates_citations(self):
        from src.xai_api import extract_tool_info

        response = MagicMock()
        response.citations = [
            "https://x.ai/news",
            "https://x.ai/news",
            "collections://collection_1/files/file_1",
        ]

        result = extract_tool_info(response)

        assert result["citations"] == [
            "https://x.ai/news",
            "collections://collection_1/files/file_1",
        ]


class TestAppendReasoningEmbeds:
    """Tests for the append_reasoning_embeds helper."""

    def test_no_reasoning(self):
        """Empty reasoning text should not add an embed."""
        from src.xai_api import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "")
        assert len(embeds) == 0

    def test_with_reasoning(self):
        """Reasoning text should be wrapped in spoiler tags."""
        from src.xai_api import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "Some reasoning here")
        assert len(embeds) == 1
        assert embeds[0].title == "Reasoning"
        assert embeds[0].description == "||Some reasoning here||"

    def test_long_reasoning_truncated(self):
        """Long reasoning text should be truncated."""
        from src.xai_api import append_reasoning_embeds

        embeds = []
        long_text = "a" * 4000
        append_reasoning_embeds(embeds, long_text)
        assert len(embeds) == 1
        assert len(embeds[0].description) < 3600
        assert "[reasoning truncated]" in embeds[0].description


class TestAppendResponseEmbeds:
    """Tests for the append_response_embeds helper."""

    def test_short_response(self):
        """Short response should create a single embed."""
        from src.xai_api import append_response_embeds

        embeds = []
        append_response_embeds(embeds, "Hello!")
        assert len(embeds) == 1
        assert embeds[0].title == "Response"
        assert embeds[0].description == "Hello!"

    def test_long_response_chunked(self):
        """Long response should be split into multiple embeds."""
        from src.xai_api import append_response_embeds

        embeds = []
        long_text = "a" * 7500
        append_response_embeds(embeds, long_text)
        assert len(embeds) > 1
        assert embeds[0].title == "Response"
        assert "Part" in embeds[1].title

    def test_very_long_response_truncated(self):
        """Very long response should be truncated before chunking."""
        from src.xai_api import append_response_embeds

        embeds = []
        very_long_text = "a" * 25000
        append_response_embeds(embeds, very_long_text)
        total_text = "".join(e.description for e in embeds)
        assert len(total_text) < 21000


class TestXAIAPICog:
    """Tests for the xAIAPI Discord cog."""

    @pytest.fixture
    def cog(self, mock_bot):
        """Create an xAIAPI cog instance with mocked dependencies."""
        with patch("xai_sdk.AsyncClient") as mock_client_class:
            mock_client = MagicMock()

            # Mock chat
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.reasoning_content = ""
            mock_response.citations = []
            mock_response.server_side_tool_usage = {}
            mock_response.tool_calls = []
            mock_chat.sample = AsyncMock(return_value=mock_response)
            mock_chat.append = MagicMock()
            mock_chat.messages = []
            mock_client.chat = MagicMock()
            mock_client.chat.create = MagicMock(return_value=mock_chat)

            mock_client_class.return_value = mock_client

            from src.xai_api import xAIAPI

            cog = xAIAPI(bot=mock_bot)
            cog.client = mock_client
            return cog

    @pytest.mark.asyncio
    async def test_cog_initialization(self, cog, mock_bot):
        """Test that the cog initializes correctly."""
        assert cog.bot == mock_bot
        assert cog.conversations == {}
        assert cog.views == {}

    @pytest.mark.asyncio
    async def test_chat_creates_conversation(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """Test that chat command creates a conversation entry."""
        cog.client = mock_xai_client

        # Mock the channel typing context manager
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello Grok!",
            model="grok-3",
        )

        # Verify chat was created and sampled
        mock_xai_client.chat.create.assert_called_once()

        # Verify conversation was stored
        assert len(cog.conversations) == 1

    @pytest.mark.asyncio
    async def test_chat_with_four_tools(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """Chat should pass the selected four tools to chat.create."""
        cog.client = mock_xai_client

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        with patch("src.xai_api.XAI_COLLECTION_IDS", ["collection_123"]):
            await cog.chat.callback(
                cog,
                ctx=mock_discord_context,
                prompt="Tool test",
                model="grok-4.20-beta-latest-reasoning",
                web_search=True,
                x_search=True,
                code_execution=True,
                collections_search=True,
            )

        create_kwargs = mock_xai_client.chat.create.call_args.kwargs
        assert "tools" in create_kwargs
        assert len(create_kwargs["tools"]) == 4
        tool_names = sorted(tool.WhichOneof("tool") for tool in create_kwargs["tools"])
        assert tool_names == [
            "code_execution",
            "collections_search",
            "web_search",
            "x_search",
        ]
        assert create_kwargs["include"] == ["inline_citations"]

    @pytest.mark.asyncio
    async def test_chat_prevents_duplicate_conversations(
        self, cog, mock_discord_context
    ):
        """Test that users can't start multiple conversations in the same channel."""
        from src.util import ChatCompletionParameters, Conversation

        existing_params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=mock_discord_context.author,
            channel_id=mock_discord_context.channel.id,
            conversation_id=123,
        )
        cog.conversations[123] = Conversation(params=existing_params, chat=MagicMock())

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello again!",
        )

        mock_discord_context.send_followup.assert_called()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "already have an active conversation" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_resolve_selected_tools_collections_requires_ids(self, cog):
        with patch("src.xai_api.XAI_COLLECTION_IDS", []):
            tools, error = cog.resolve_selected_tools(["collections_search"])

        assert tools == []
        assert "XAI_COLLECTION_IDS" in error

    @pytest.mark.asyncio
    async def test_resolve_selected_tools_success(self, cog):
        with patch("src.xai_api.XAI_COLLECTION_IDS", ["collection_abc"]):
            tools, error = cog.resolve_selected_tools(
                ["web_search", "x_search", "code_execution", "collections_search"]
            )

        assert error is None
        assert len(tools) == 4
        tool_names = sorted(tool.WhichOneof("tool") for tool in tools)
        assert tool_names == [
            "code_execution",
            "collections_search",
            "web_search",
            "x_search",
        ]

    @pytest.mark.asyncio
    async def test_resolve_selected_tools_x_search_with_kwargs(self, cog):
        """x_search kwargs should be forwarded to the x_search tool builder."""
        x_search_kw = {
            "enable_image_understanding": True,
            "allowed_x_handles": ["elonmusk"],
        }
        tools, error = cog.resolve_selected_tools(
            ["x_search"], x_search_kwargs=x_search_kw
        )

        assert error is None
        assert len(tools) == 1
        assert tools[0].WhichOneof("tool") == "x_search"

    @pytest.mark.asyncio
    async def test_resolve_selected_tools_x_search_without_kwargs(self, cog):
        """x_search without kwargs should still work."""
        tools, error = cog.resolve_selected_tools(["x_search"])

        assert error is None
        assert len(tools) == 1
        assert tools[0].WhichOneof("tool") == "x_search"

    @pytest.mark.asyncio
    async def test_resolve_selected_tools_web_search_with_kwargs(self, cog):
        """web_search kwargs should be forwarded to the web_search tool builder."""
        web_search_kw = {
            "enable_image_understanding": True,
            "allowed_domains": ["example.com"],
        }
        tools, error = cog.resolve_selected_tools(
            ["web_search"], web_search_kwargs=web_search_kw
        )

        assert error is None
        assert len(tools) == 1
        assert tools[0].WhichOneof("tool") == "web_search"

    @pytest.mark.asyncio
    async def test_resolve_selected_tools_web_search_without_kwargs(self, cog):
        """web_search without kwargs should still work."""
        tools, error = cog.resolve_selected_tools(["web_search"])

        assert error is None
        assert len(tools) == 1
        assert tools[0].WhichOneof("tool") == "web_search"

    @pytest.mark.asyncio
    async def test_chat_default_model(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """Chat should use grok-4.20-beta-latest-reasoning as the default model."""
        cog.client = mock_xai_client

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello!",
        )

        create_kwargs = mock_xai_client.chat.create.call_args.kwargs
        assert create_kwargs["model"] == "grok-4.20-beta-latest-reasoning"

    @pytest.mark.asyncio
    async def test_chat_rejects_frequency_penalty_on_reasoning_model(
        self, cog, mock_discord_context
    ):
        """frequency_penalty should be rejected for reasoning models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3-mini",
            frequency_penalty=0.5,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "frequency_penalty" in call_kwargs["embed"].description
        assert "not supported" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_chat_rejects_both_penalties_on_reasoning_model(
        self, cog, mock_discord_context
    ):
        """Both penalties set on a reasoning model should be rejected."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3",
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "frequency_penalty" in call_kwargs["embed"].description
        assert "presence_penalty" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_chat_allows_penalty_on_non_reasoning_model(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """Penalty params should be allowed for non-reasoning models."""
        cog.client = mock_xai_client

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-beta-latest-non-reasoning",
            frequency_penalty=0.5,
        )

        create_kwargs = mock_xai_client.chat.create.call_args.kwargs
        assert create_kwargs["frequency_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_chat_rejects_reasoning_effort_on_unsupported_model(
        self, cog, mock_discord_context
    ):
        """reasoning_effort should be rejected for models that don't support it."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3",
            reasoning_effort="high",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "reasoning_effort" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_chat_passes_reasoning_effort_for_supported_model(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """reasoning_effort should be passed to the API for grok-3-mini."""
        cog.client = mock_xai_client

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3-mini",
            reasoning_effort="high",
        )

        create_kwargs = mock_xai_client.chat.create.call_args.kwargs
        assert create_kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_chat_rejects_max_tokens_on_multi_agent(
        self, cog, mock_discord_context
    ):
        """max_tokens should be rejected for multi-agent models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-multi-agent-beta-latest",
            max_tokens=1024,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "max_tokens" in call_kwargs["embed"].description
        assert "not supported" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_chat_rejects_agent_count_on_non_multi_agent(
        self, cog, mock_discord_context
    ):
        """agent_count should be rejected for non-multi-agent models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-3",
            agent_count=4,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "agent_count" in call_kwargs["embed"].description
        assert "multi-agent" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_chat_passes_agent_count_for_multi_agent(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """agent_count should be passed to chat.create for multi-agent models."""
        cog.client = mock_xai_client

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Research quantum computing",
            model="grok-4.20-multi-agent-beta-latest",
            agent_count=16,
        )

        create_kwargs = mock_xai_client.chat.create.call_args.kwargs
        assert create_kwargs["agent_count"] == 16
        assert create_kwargs["use_encrypted_content"] is True

    @pytest.mark.asyncio
    async def test_chat_multi_agent_sets_encrypted_content(
        self, cog, mock_discord_context, mock_xai_client
    ):
        """Multi-agent model should always set use_encrypted_content=True."""
        cog.client = mock_xai_client

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-multi-agent-beta-latest",
        )

        create_kwargs = mock_xai_client.chat.create.call_args.kwargs
        assert create_kwargs["use_encrypted_content"] is True
        assert "agent_count" not in create_kwargs

    def test_chat_model_choices_match_grok_models(self, cog):
        """Chat command model choices should match GROK_MODELS."""
        from src.util import GROK_MODELS

        # Extract choice values from the chat command's model option
        chat_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "chat"
        )

        model_option = next(
            opt for opt in chat_cmd.options if opt.name == "model"
        )
        choice_values = sorted(c.value for c in model_option.choices)
        assert choice_values == sorted(GROK_MODELS)

    def test_image_model_choices_match_grok_image_models(self, cog):
        """Image command model choices should match GROK_IMAGE_MODELS."""
        from src.util import GROK_IMAGE_MODELS

        image_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "image"
        )

        model_option = next(
            opt for opt in image_cmd.options if opt.name == "model"
        )
        choice_values = sorted(c.value for c in model_option.choices)
        assert choice_values == sorted(GROK_IMAGE_MODELS)

    @pytest.mark.asyncio
    async def test_on_message_ignores_bot_messages(self, cog, mock_discord_message):
        """Test that the bot ignores its own messages."""
        mock_discord_message.author = cog.bot.user

        await cog.on_message(mock_discord_message)

        mock_discord_message.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_keep_typing_can_be_cancelled(self, cog, mock_discord_context):
        """Test that the typing indicator can be cancelled."""
        import asyncio

        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=None)
        mock_discord_context.channel.typing = MagicMock(return_value=typing_cm)

        task = asyncio.create_task(cog.keep_typing(mock_discord_context.channel))

        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestTTSCommand:
    """Tests for the /grok tts command."""

    @pytest.fixture
    def cog(self, mock_bot):
        with patch("xai_sdk.AsyncClient"):
            from src.xai_api import xAIAPI

            cog = xAIAPI(bot=mock_bot)
            return cog

    @pytest.mark.asyncio
    async def test_tts_text_too_long(self, cog, mock_discord_context):
        """Text over 15,000 chars should be rejected."""
        await cog.tts.callback(
            cog,
            ctx=mock_discord_context,
            text="a" * 15001,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "15,000" in call_kwargs["embed"].description

    @pytest.mark.asyncio
    async def test_tts_success(self, cog, mock_discord_context):
        """Successful TTS should send an audio file with metadata embed."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"fake audio bytes"

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hello world",
                voice="eve",
                language="en",
                output_format="mp3",
            )

        mock_gen.assert_awaited_once_with("Hello world", "eve", "en", "mp3")
        mock_discord_context.send_followup.assert_called_once()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embed"].title == "Text-to-Speech Generation"
        assert call_kwargs["file"] is not None

    @pytest.mark.asyncio
    async def test_tts_api_error(self, cog, mock_discord_context):
        """API errors should display an error embed."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("TTS API error (HTTP 400): bad request")

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hello",
            )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embed"].title == "Error"

    def test_tts_voice_choices_match_tts_voices(self, cog):
        """TTS command voice choices should match TTS_VOICES."""
        from src.util import TTS_VOICES

        tts_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "tts")
        voice_option = next(opt for opt in tts_cmd.options if opt.name == "voice")
        choice_values = sorted(c.value for c in voice_option.choices)
        assert choice_values == sorted(TTS_VOICES)


class TestFileUploadAndCleanup:
    """Tests for the xAI Files API integration."""

    @pytest.fixture
    def cog(self, mock_bot):
        """Create an xAIAPI cog instance with mocked dependencies."""
        with patch("xai_sdk.AsyncClient") as mock_client_class:
            mock_client = MagicMock()

            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.reasoning_content = ""
            mock_response.citations = []
            mock_chat.sample = AsyncMock(return_value=mock_response)
            mock_chat.append = MagicMock()
            mock_chat.messages = []
            mock_client.chat = MagicMock()
            mock_client.chat.create = MagicMock(return_value=mock_chat)

            # Mock files API
            mock_uploaded_file = MagicMock()
            mock_uploaded_file.id = "file-abc123"
            mock_uploaded_file.filename = "document.pdf"
            mock_client.files = MagicMock()
            mock_client.files.upload = AsyncMock(return_value=mock_uploaded_file)
            mock_client.files.delete = AsyncMock()

            mock_client_class.return_value = mock_client

            from src.xai_api import xAIAPI

            cog = xAIAPI(bot=mock_bot)
            cog.client = mock_client
            return cog

    @pytest.mark.asyncio
    async def test_upload_file_attachment_success(self, cog, mock_file_attachment):
        """Should download from Discord and upload to xAI, returning the file ID."""
        with patch.object(cog, "_fetch_attachment_bytes", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = b"file content"

            file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id == "file-abc123"
        cog.client.files.upload.assert_awaited_once_with(
            b"file content", filename="document.pdf"
        )

    @pytest.mark.asyncio
    async def test_upload_file_attachment_too_large(self, cog, mock_file_attachment):
        """Files exceeding 48 MB should be rejected."""
        mock_file_attachment.size = 50 * 1024 * 1024  # 50 MB

        file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id is None
        cog.client.files.upload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_file_attachment_fetch_fails(self, cog, mock_file_attachment):
        """Should return None when the Discord download fails."""
        with patch.object(cog, "_fetch_attachment_bytes", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id is None
        cog.client.files.upload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_file_attachment_xai_upload_fails(self, cog, mock_file_attachment):
        """Should return None when the xAI upload fails."""
        with patch.object(cog, "_fetch_attachment_bytes", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = b"file content"
            cog.client.files.upload.side_effect = Exception("Upload failed")

            file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id is None

    @pytest.mark.asyncio
    async def test_cleanup_conversation_files(self, cog):
        """Should delete all tracked file IDs from xAI."""
        from src.util import ChatCompletionParameters, Conversation

        conversation = Conversation(
            params=ChatCompletionParameters(model="grok-3"),
            chat=MagicMock(),
            file_ids=["file-1", "file-2", "file-3"],
        )

        await cog._cleanup_conversation_files(conversation)

        assert cog.client.files.delete.await_count == 3
        cog.client.files.delete.assert_any_await("file-1")
        cog.client.files.delete.assert_any_await("file-2")
        cog.client.files.delete.assert_any_await("file-3")
        assert conversation.file_ids == []

    @pytest.mark.asyncio
    async def test_cleanup_continues_on_failure(self, cog):
        """Should continue deleting remaining files even if one fails."""
        from src.util import ChatCompletionParameters, Conversation

        conversation = Conversation(
            params=ChatCompletionParameters(model="grok-3"),
            chat=MagicMock(),
            file_ids=["file-1", "file-2"],
        )
        cog.client.files.delete.side_effect = [Exception("Failed"), None]

        await cog._cleanup_conversation_files(conversation)

        assert cog.client.files.delete.await_count == 2
        assert conversation.file_ids == []

    @pytest.mark.asyncio
    async def test_end_conversation_cleans_up_files(self, cog):
        """end_conversation should remove the conversation and delete files."""
        from src.util import ChatCompletionParameters, Conversation

        conversation = Conversation(
            params=ChatCompletionParameters(model="grok-3"),
            chat=MagicMock(),
            file_ids=["file-1"],
        )
        cog.conversations[999] = conversation

        await cog.end_conversation(999)

        assert 999 not in cog.conversations
        cog.client.files.delete.assert_awaited_once_with("file-1")

    @pytest.mark.asyncio
    async def test_end_conversation_missing_id(self, cog):
        """end_conversation with unknown ID should not error."""
        await cog.end_conversation(999)
        cog.client.files.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chat_with_file_attachment(
        self, cog, mock_discord_context, mock_xai_client, mock_file_attachment
    ):
        """Chat command with a non-image attachment should upload to xAI Files API."""
        cog.client = mock_xai_client

        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        with patch.object(cog, "_upload_file_attachment", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = "file-xyz789"

            await cog.chat.callback(
                cog,
                ctx=mock_discord_context,
                prompt="What does this file say?",
                model="grok-3",
                attachment=mock_file_attachment,
            )

        mock_upload.assert_awaited_once_with(mock_file_attachment)
        assert len(cog.conversations) == 1
        conversation = list(cog.conversations.values())[0]
        assert "file-xyz789" in conversation.file_ids
