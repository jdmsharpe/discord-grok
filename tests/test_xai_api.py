import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import MOCK_RESPONSES_API_RESPONSE
from discord import Colour, Embed


class TestAppendPricingEmbed:
    """Tests for the append_pricing_embed helper."""

    def test_append_pricing_embed(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50)
        assert len(embeds) == 1
        desc = embeds[0].description
        assert "1,000 tokens in" in desc
        assert "500 tokens out" in desc
        assert "daily $1.50" in desc
        assert embeds[0].colour == Colour(0)

    def test_append_pricing_embed_with_reasoning_tokens(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, reasoning_tokens=200)
        assert len(embeds) == 1
        assert "200 reasoning" in embeds[0].description

    def test_append_pricing_embed_hides_zero_reasoning_tokens(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, reasoning_tokens=0)
        assert "reasoning" not in embeds[0].description

    def test_append_pricing_embed_with_cached_tokens(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, cached_tokens=300)
        assert "300 cached" in embeds[0].description

    def test_append_pricing_embed_with_image_tokens(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, image_tokens=200)
        assert "200 image" in embeds[0].description

    def test_append_pricing_embed_hides_zero_cached_and_image_tokens(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, cached_tokens=0, image_tokens=0)
        assert "cached" not in embeds[0].description
        assert "image" not in embeds[0].description

    def test_append_pricing_embed_with_tool_usage(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        tool_usage = {"SERVER_SIDE_TOOL_WEB_SEARCH": 3, "SERVER_SIDE_TOOL_X_SEARCH": 2}
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, tool_usage=tool_usage)
        desc = embeds[0].description
        assert desc is not None
        assert "Web Search \u00d73" in desc
        assert "X Search \u00d72" in desc
        assert "\n" in desc
        assert "tool cost" in desc

    def test_append_pricing_embed_no_tool_usage_line(self):
        from xai_api import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, tool_usage={})
        assert "\n" not in embeds[0].description

    def test_append_generation_pricing_embed(self):
        from xai_api import append_generation_pricing_embed

        embeds: list[Embed] = []
        append_generation_pricing_embed(embeds, 0.07, 2.50)
        assert len(embeds) == 1
        assert "$0.0700" in embeds[0].description
        assert "daily $2.50" in embeds[0].description


class TestTrackDailyCost:
    """Tests for the _track_daily_cost method."""

    @pytest.fixture
    def cog(self, mock_bot):
        with patch("xai_sdk.AsyncClient"):
            from xai_api import xAIAPI

            cog = xAIAPI(bot=mock_bot)
            return cog

    def test_track_daily_cost_accumulates(self, cog):
        daily = cog._track_daily_cost(1, 3.00)
        assert daily == pytest.approx(3.00)
        daily = cog._track_daily_cost(1, 15.00)
        assert daily == pytest.approx(18.00)

    def test_track_daily_cost_isolates_users(self, cog):
        cog._track_daily_cost(1, 5.00)
        daily = cog._track_daily_cost(2, 3.00)
        assert daily == pytest.approx(3.00)


class TestExtractToolInfo:
    """Tests for extract_tool_info helper."""

    def test_annotations_deduplicates_and_classifies_citations(self):
        """URL citations in annotations should be deduplicated and classified."""
        from xai_api import extract_tool_info

        response_json = {
            "output": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Some text",
                            "annotations": [
                                {"type": "url_citation", "url": "https://x.ai/news"},
                                {"type": "url_citation", "url": "https://x.ai/news"},
                                {"type": "url_citation", "url": "https://x.com/i/status/123"},
                                {"type": "url_citation", "url": "collections://collection_1/files/file_1"},
                            ],
                        }
                    ],
                }
            ]
        }

        result = extract_tool_info(response_json)

        assert result["citations"] == [
            {"url": "https://x.ai/news", "source": "web"},
            {"url": "https://x.com/i/status/123", "source": "x"},
            {"url": "collections://collection_1/files/file_1", "source": "collections"},
        ]

    def test_annotations_web_and_x(self):
        """Mixed web and X citations should be classified correctly."""
        from xai_api import extract_tool_info

        response_json = {
            "output": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Some text",
                            "annotations": [
                                {"type": "url_citation", "url": "https://example.com/article"},
                                {"type": "url_citation", "url": "https://x.com/i/status/456"},
                            ],
                        }
                    ],
                }
            ]
        }

        result = extract_tool_info(response_json)

        assert result["citations"] == [
            {"url": "https://example.com/article", "source": "web"},
            {"url": "https://x.com/i/status/456", "source": "x"},
        ]

    def test_empty_output(self):
        """Empty output should return no citations."""
        from xai_api import extract_tool_info

        result = extract_tool_info({"output": []})
        assert result["citations"] == []

    def test_no_annotations(self):
        """Output without annotations should return no citations."""
        from xai_api import extract_tool_info

        response_json = {
            "output": [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "No citations here"}],
                }
            ]
        }
        result = extract_tool_info(response_json)
        assert result["citations"] == []


class TestAppendReasoningEmbeds:
    """Tests for the append_reasoning_embeds helper."""

    def test_no_reasoning(self):
        """Empty reasoning text should not add an embed."""
        from xai_api import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "")
        assert len(embeds) == 0

    def test_with_reasoning(self):
        """Reasoning text should be wrapped in spoiler tags."""
        from xai_api import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "Some reasoning here")
        assert len(embeds) == 1
        assert embeds[0].title == "Reasoning"
        assert embeds[0].description == "||Some reasoning here||"

    def test_long_reasoning_truncated(self):
        """Long reasoning text should be truncated."""
        from xai_api import append_reasoning_embeds

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
        from xai_api import append_response_embeds

        embeds = []
        append_response_embeds(embeds, "Hello!")
        assert len(embeds) == 1
        assert embeds[0].title == "Response"
        assert embeds[0].description == "Hello!"

    def test_long_response_chunked(self):
        """Long response should be split into multiple embeds."""
        from xai_api import append_response_embeds

        embeds = []
        long_text = "a" * 7500
        append_response_embeds(embeds, long_text)
        assert len(embeds) > 1
        assert embeds[0].title == "Response"
        assert "Part" in embeds[1].title

    def test_very_long_response_truncated(self):
        """Very long response should be truncated before chunking."""
        from xai_api import append_response_embeds

        embeds = []
        very_long_text = "a" * 25000
        append_response_embeds(embeds, very_long_text)
        total_text = "".join(e.description for e in embeds)
        assert len(total_text) < 21000


def _make_cog(mock_bot, mock_api_response=None):
    """Helper to create a cog with _call_responses_api mocked."""
    with patch("xai_sdk.AsyncClient"):
        from xai_api import xAIAPI

        cog = xAIAPI(bot=mock_bot)

    if mock_api_response is None:
        mock_api_response = copy.deepcopy(MOCK_RESPONSES_API_RESPONSE)
    cog._call_responses_api = AsyncMock(return_value=mock_api_response)
    return cog


class TestXAIAPICog:
    """Tests for the xAIAPI Discord cog."""

    @pytest.fixture
    def cog(self, mock_bot):
        return _make_cog(mock_bot)

    async def test_cog_initialization(self, cog, mock_bot):
        """Test that the cog initializes correctly."""
        assert cog.bot == mock_bot
        assert cog.conversations == {}
        assert cog.views == {}

    async def test_chat_creates_conversation(self, cog, mock_discord_context):
        """Test that chat command creates a conversation entry."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello Grok!",
            model="grok-3",
        )

        cog._call_responses_api.assert_called_once()
        assert len(cog.conversations) == 1

        # Verify payload structure
        payload = cog._call_responses_api.call_args[0][0]
        assert payload["model"] == "grok-3"
        assert payload["store"] is True
        assert any(
            msg.get("role") == "user"
            for msg in payload["input"]
        )

    async def test_chat_stores_response_id(self, cog, mock_discord_context):
        """Chat should store the response ID for multi-turn."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello!",
            model="grok-3",
        )

        conversation = list(cog.conversations.values())[0]
        assert conversation.previous_response_id == "resp_01XFDUDYJgAACzvnptvVoYEL"
        assert conversation.response_id_history == ["resp_01XFDUDYJgAACzvnptvVoYEL"]

    async def test_chat_with_four_tools(self, cog, mock_discord_context):
        """Chat should pass the selected four tools in the payload."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        with patch("xai_api.XAI_COLLECTION_IDS", ["collection_123"]):
            await cog.chat.callback(
                cog,
                ctx=mock_discord_context,
                prompt="Tool test",
                model="grok-4.20",
                web_search=True,
                x_search=True,
                code_execution=True,
                collections_search=True,
            )

        payload = cog._call_responses_api.call_args[0][0]
        assert "tools" in payload
        assert len(payload["tools"]) == 4
        tool_types = sorted(t["type"] for t in payload["tools"])
        assert tool_types == [
            "code_interpreter",
            "file_search",
            "web_search",
            "x_search",
        ]
        # Tools should trigger encrypted content include
        assert "reasoning.encrypted_content" in payload.get("include", [])

    async def test_chat_prevents_duplicate_conversations(
        self, cog, mock_discord_context
    ):
        """Test that users can't start multiple conversations in the same channel."""
        from util import ChatCompletionParameters, Conversation

        existing_params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=mock_discord_context.author,
            channel_id=mock_discord_context.channel.id,
            conversation_id=123,
        )
        cog.conversations[123] = Conversation(params=existing_params)

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello again!",
        )

        mock_discord_context.send_followup.assert_called()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "already have an active conversation" in call_kwargs["embed"].description

    async def test_resolve_selected_tools_collections_requires_ids(self, cog):
        with patch("xai_api.XAI_COLLECTION_IDS", []):
            tools, error = cog.resolve_selected_tools(["collections_search"])

        assert tools == []
        assert "XAI_COLLECTION_IDS" in error

    async def test_resolve_selected_tools_success(self, cog):
        with patch("xai_api.XAI_COLLECTION_IDS", ["collection_abc"]):
            tools, error = cog.resolve_selected_tools(
                ["web_search", "x_search", "code_execution", "collections_search"]
            )

        assert error is None
        assert len(tools) == 4
        tool_types = sorted(t["type"] for t in tools)
        assert tool_types == [
            "code_interpreter",
            "file_search",
            "web_search",
            "x_search",
        ]

    async def test_resolve_selected_tools_x_search_with_kwargs(self, cog):
        """x_search kwargs should be included in the tool dict."""
        x_search_kw = {
            "enable_image_understanding": True,
            "allowed_x_handles": ["elonmusk"],
        }
        tools, error = cog.resolve_selected_tools(
            ["x_search"], x_search_kwargs=x_search_kw
        )

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "x_search"
        assert tools[0]["enable_image_understanding"] is True
        assert tools[0]["allowed_x_handles"] == ["elonmusk"]

    async def test_resolve_selected_tools_x_search_without_kwargs(self, cog):
        """x_search without kwargs should still work."""
        tools, error = cog.resolve_selected_tools(["x_search"])

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "x_search"

    async def test_resolve_selected_tools_web_search_with_kwargs(self, cog):
        """web_search kwargs should be included in the tool dict."""
        web_search_kw = {
            "enable_image_understanding": True,
            "allowed_domains": ["example.com"],
        }
        tools, error = cog.resolve_selected_tools(
            ["web_search"], web_search_kwargs=web_search_kw
        )

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search"
        assert tools[0]["enable_image_understanding"] is True
        assert tools[0]["allowed_domains"] == ["example.com"]

    async def test_resolve_selected_tools_web_search_without_kwargs(self, cog):
        """web_search without kwargs should still work."""
        tools, error = cog.resolve_selected_tools(["web_search"])

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search"

    async def test_resolve_selected_tools_passes_date_strings(self, cog):
        """ISO date strings in x_search kwargs should be passed through."""
        x_search_kw = {
            "from_date": "2024-01-01T00:00:00",
            "to_date": "2024-12-31T00:00:00",
        }
        tools, error = cog.resolve_selected_tools(
            ["x_search"], x_search_kwargs=x_search_kw
        )

        assert error is None
        assert tools[0]["from_date"] == "2024-01-01T00:00:00"
        assert tools[0]["to_date"] == "2024-12-31T00:00:00"

    async def test_chat_default_model(self, cog, mock_discord_context):
        """Chat should use grok-4.20 as the default model."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello!",
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["model"] == "grok-4.20"

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

    async def test_chat_allows_penalty_on_non_reasoning_model(
        self, cog, mock_discord_context
    ):
        """Penalty params should be allowed for non-reasoning models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-non-reasoning",
            frequency_penalty=0.5,
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["frequency_penalty"] == 0.5

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

    async def test_chat_passes_reasoning_effort_for_supported_model(
        self, cog, mock_discord_context
    ):
        """reasoning_effort should be passed to the API for grok-3-mini."""
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

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["reasoning_effort"] == "high"

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
            model="grok-4.20-multi-agent",
            max_tokens=1024,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "max_tokens" in call_kwargs["embed"].description
        assert "not supported" in call_kwargs["embed"].description

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

    async def test_chat_passes_agent_count_for_multi_agent(
        self, cog, mock_discord_context
    ):
        """agent_count should be passed to the API for multi-agent models."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Research quantum computing",
            model="grok-4.20-multi-agent",
            agent_count=16,
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["agent_count"] == 16
        assert "reasoning.encrypted_content" in payload["include"]

    async def test_chat_multi_agent_sets_encrypted_content(
        self, cog, mock_discord_context
    ):
        """Multi-agent model should always set include with encrypted content."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Hello",
            model="grok-4.20-multi-agent",
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert "reasoning.encrypted_content" in payload["include"]
        assert "agent_count" not in payload

    async def test_chat_tools_set_encrypted_content(
        self, cog, mock_discord_context
    ):
        """Tool-using conversations should include encrypted content."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Search for news",
            model="grok-4.20",
            web_search=True,
        )

        payload = cog._call_responses_api.call_args[0][0]
        assert "reasoning.encrypted_content" in payload["include"]

    def test_chat_model_choices_match_grok_models(self, cog):
        """Chat command model choices should match GROK_MODELS."""
        from util import GROK_MODELS

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
        from util import GROK_IMAGE_MODELS

        image_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "image"
        )

        model_option = next(
            opt for opt in image_cmd.options if opt.name == "model"
        )
        choice_values = sorted(c.value for c in model_option.choices)
        assert choice_values == sorted(GROK_IMAGE_MODELS)

    def test_image_aspect_ratios_match_sdk(self, cog):
        """Image command aspect ratios should match xai-sdk ImageAspectRatio."""
        from xai_sdk.image import ImageAspectRatio

        image_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "image"
        )
        ar_option = next(
            opt for opt in image_cmd.options if opt.name == "aspect_ratio"
        )
        choice_values = sorted(c.value for c in ar_option.choices)
        assert choice_values == sorted(ImageAspectRatio.__args__)

    def test_image_resolution_choices_match_sdk(self, cog):
        """Image command resolution choices should match xai-sdk ImageResolution."""
        from xai_sdk.image import ImageResolution

        image_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "image"
        )
        res_option = next(
            opt for opt in image_cmd.options if opt.name == "resolution"
        )
        choice_values = sorted(c.value for c in res_option.choices)
        assert choice_values == sorted(ImageResolution.__args__)

    def test_image_has_attachment_option(self, cog):
        """Image command should have an optional attachment parameter."""
        image_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "image"
        )
        att_option = next(
            (opt for opt in image_cmd.options if opt.name == "attachment"), None
        )
        assert att_option is not None
        assert att_option.required is False

    def test_video_aspect_ratios_match_sdk(self, cog):
        """Video command aspect ratios should match xai-sdk VideoAspectRatio."""
        from xai_sdk.video import VideoAspectRatio

        video_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "video"
        )
        ar_option = next(
            opt for opt in video_cmd.options if opt.name == "aspect_ratio"
        )
        choice_values = sorted(c.value for c in ar_option.choices)
        assert choice_values == sorted(VideoAspectRatio.__args__)

    def test_video_has_attachment_option(self, cog):
        """Video command should have an optional attachment parameter."""
        video_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "video"
        )
        att_option = next(
            (opt for opt in video_cmd.options if opt.name == "attachment"), None
        )
        assert att_option is not None
        assert att_option.required is False

    async def test_on_message_ignores_bot_messages(self, cog, mock_discord_message):
        """Test that the bot ignores its own messages."""
        mock_discord_message.author = cog.bot.user

        await cog.on_message(mock_discord_message)

        mock_discord_message.reply.assert_not_called()

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


class TestResponseParsing:
    """Tests for response parsing helpers."""

    @pytest.fixture
    def cog(self, mock_bot):
        return _make_cog(mock_bot)

    def test_extract_response_text_basic(self, cog):
        text, reasoning = cog._extract_response_text(MOCK_RESPONSES_API_RESPONSE)
        assert text == "Hello! How can I help you today?"
        assert reasoning == ""

    def test_extract_response_text_strips_citation_markers(self, cog):
        response = {
            "output": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "xAI is building AI [[1]](https://x.ai/) systems.",
                        }
                    ],
                }
            ]
        }
        text, _ = cog._extract_response_text(response)
        assert "[[1]]" not in text
        assert "xAI is building AI" in text

    def test_extract_response_text_with_reasoning(self, cog):
        response = {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "Thinking step 1."},
                        {"type": "summary_text", "text": " Thinking step 2."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The answer is 42."},
                    ],
                },
            ]
        }
        text, reasoning = cog._extract_response_text(response)
        assert text == "The answer is 42."
        assert reasoning == "Thinking step 1. Thinking step 2."

    def test_extract_usage(self, cog):
        usage = cog._extract_usage(MOCK_RESPONSES_API_RESPONSE)
        assert usage["input_tokens"] == 25
        assert usage["output_tokens"] == 50
        assert usage["reasoning_tokens"] == 0
        assert usage["cached_tokens"] == 0
        assert usage["image_tokens"] == 0

    def test_extract_usage_with_details(self, cog):
        """Responses API field names (input_tokens_details, output_tokens_details)."""
        response = {
            "usage": {
                "input_tokens": 199,
                "output_tokens": 530,
                "input_tokens_details": {
                    "cached_tokens": 163,
                    "image_tokens": 50,
                },
                "output_tokens_details": {
                    "reasoning_tokens": 310,
                },
            }
        }
        usage = cog._extract_usage(response)
        assert usage["input_tokens"] == 199
        assert usage["output_tokens"] == 530
        assert usage["reasoning_tokens"] == 310

    def test_extract_usage_fallback_field_names(self, cog):
        """Chat Completions field names should still work as fallback."""
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "prompt_tokens_details": {
                    "cached_tokens": 50,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 30,
                },
            }
        }
        usage = cog._extract_usage(response)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 200
        assert usage["reasoning_tokens"] == 30
        assert usage["cached_tokens"] == 50

    def test_build_user_message_text_only(self, cog):
        msg = cog._build_user_message(["Hello!"])
        assert msg == {"role": "user", "content": "Hello!"}

    def test_build_user_message_multimodal(self, cog):
        msg = cog._build_user_message([
            "Describe this",
            {"type": "input_image", "image_url": "https://example.com/img.png", "detail": "high"},
        ])
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "input_text", "text": "Describe this"}
        assert msg["content"][1]["type"] == "input_image"


class TestTTSCommand:
    """Tests for the /grok tts command."""

    @pytest.fixture
    def cog(self, mock_bot):
        with patch("xai_sdk.AsyncClient"):
            from xai_api import xAIAPI

            cog = xAIAPI(bot=mock_bot)
            return cog

    async def test_tts_text_too_long(self, cog, mock_discord_context):
        """Text over 15,000 chars should be rejected."""
        await cog.tts.callback(
            cog,
            ctx=mock_discord_context,
            text="a" * 15001,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "15,000" in call_kwargs["embed"].description

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

        mock_gen.assert_awaited_once_with("Hello world", "eve", "en", "mp3", None, None)
        mock_discord_context.send_followup.assert_called_once()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embeds"][0].title == "Text-to-Speech Generation"
        assert call_kwargs["file"] is not None

    async def test_tts_with_sample_rate_and_bit_rate(self, cog, mock_discord_context):
        """sample_rate and bit_rate should be forwarded to _generate_tts."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"fake audio bytes"

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hi",
                voice="rex",
                language="auto",
                output_format="mp3",
                sample_rate=44100,
                bit_rate=192000,
            )

        mock_gen.assert_awaited_once_with("Hi", "rex", "auto", "mp3", 44100, 192000)
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "44,100 Hz" in call_kwargs["embeds"][0].description
        assert "192 kbps" in call_kwargs["embeds"][0].description

    async def test_tts_mulaw_file_extension(self, cog, mock_discord_context):
        """mulaw codec should produce a .ulaw file extension."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"fake audio bytes"

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hello",
                output_format="mulaw",
            )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["file"].filename == "speech.ulaw"

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
        from util import TTS_VOICES

        tts_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "tts")
        voice_option = next(opt for opt in tts_cmd.options if opt.name == "voice")
        choice_values = sorted(c.value for c in voice_option.choices)
        assert choice_values == sorted(TTS_VOICES)


class TestFileUploadAndCleanup:
    """Tests for the xAI Files API integration."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client):
        """Create a cog with files API mocked."""
        cog = _make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    async def test_upload_file_attachment_success(self, cog, mock_file_attachment):
        """Should download from Discord and upload to xAI, returning the file ID."""
        with patch.object(cog, "_fetch_attachment_bytes", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = b"file content"

            file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id == "file-abc123"
        cog.client.files.upload.assert_awaited_once_with(
            b"file content", filename="document.pdf"
        )

    async def test_upload_file_attachment_too_large(self, cog, mock_file_attachment):
        """Files exceeding 48 MB should be rejected."""
        mock_file_attachment.size = 50 * 1024 * 1024  # 50 MB

        file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id is None
        cog.client.files.upload.assert_not_awaited()

    async def test_upload_file_attachment_fetch_fails(self, cog, mock_file_attachment):
        """Should return None when the Discord download fails."""
        with patch.object(cog, "_fetch_attachment_bytes", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id is None
        cog.client.files.upload.assert_not_awaited()

    async def test_upload_file_attachment_xai_upload_fails(self, cog, mock_file_attachment):
        """Should return None when the xAI upload fails."""
        with patch.object(cog, "_fetch_attachment_bytes", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = b"file content"
            cog.client.files.upload.side_effect = Exception("Upload failed")

            file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id is None

    async def test_cleanup_conversation_files(self, cog):
        """Should delete all tracked file IDs from xAI."""
        from util import ChatCompletionParameters, Conversation

        conversation = Conversation(
            params=ChatCompletionParameters(model="grok-3"),
            file_ids=["file-1", "file-2", "file-3"],
        )

        await cog._cleanup_conversation_files(conversation)

        assert cog.client.files.delete.await_count == 3
        cog.client.files.delete.assert_any_await("file-1")
        cog.client.files.delete.assert_any_await("file-2")
        cog.client.files.delete.assert_any_await("file-3")
        assert conversation.file_ids == []

    async def test_cleanup_continues_on_failure(self, cog):
        """Should continue deleting remaining files even if one fails."""
        from util import ChatCompletionParameters, Conversation

        conversation = Conversation(
            params=ChatCompletionParameters(model="grok-3"),
            file_ids=["file-1", "file-2"],
        )
        cog.client.files.delete.side_effect = [Exception("Failed"), None]

        await cog._cleanup_conversation_files(conversation)

        assert cog.client.files.delete.await_count == 2
        assert conversation.file_ids == []

    async def test_end_conversation_cleans_up_files(self, cog):
        """end_conversation should remove the conversation and delete files."""
        from util import ChatCompletionParameters, Conversation

        conversation = Conversation(
            params=ChatCompletionParameters(model="grok-3"),
            file_ids=["file-1"],
        )
        cog.conversations[999] = conversation

        await cog.end_conversation(999)

        assert 999 not in cog.conversations
        cog.client.files.delete.assert_awaited_once_with("file-1")

    async def test_end_conversation_missing_id(self, cog):
        """end_conversation with unknown ID should not error."""
        await cog.end_conversation(999)
        cog.client.files.delete.assert_not_awaited()

    async def test_chat_with_file_attachment(
        self, cog, mock_discord_context, mock_file_attachment
    ):
        """Chat command with a non-image attachment should upload to xAI Files API."""
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


class TestSessionManagement:
    """Tests for shared aiohttp session and timeout configuration."""

    @pytest.fixture
    def cog(self, mock_bot):
        return _make_cog(mock_bot)

    async def test_get_http_session_has_timeout(self, cog):
        """Shared session should be created with explicit timeouts."""
        session = await cog._get_http_session()
        assert session.timeout.total == 300
        assert session.timeout.connect == 15
        await session.close()

    async def test_get_http_session_reuses_session(self, cog):
        """Calling _get_http_session twice should return the same session."""
        session1 = await cog._get_http_session()
        session2 = await cog._get_http_session()
        assert session1 is session2
        await session1.close()


class TestImageBatchGeneration:
    """Tests for multi-image generation via sample_batch."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client):
        """Create a cog with xAI image SDK mocked."""
        cog = _make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    def test_image_has_count_option(self, cog):
        """Image command should have a count parameter with min=1, max=10."""
        image_cmd = next(
            cmd for cmd in cog.grok.walk_commands() if cmd.name == "image"
        )
        count_option = next(
            (opt for opt in image_cmd.options if opt.name == "count"), None
        )
        assert count_option is not None
        assert count_option.required is False
        assert count_option.min_value == 1
        assert count_option.max_value == 10

    @staticmethod
    def _mock_http_session():
        """Create a mock HTTP session with a working async context manager for get()."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"fake image bytes")

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm
        return mock_session

    async def test_image_single_calls_sample(self, cog, mock_discord_context):
        """count=1 should call client.image.sample (not sample_batch)."""
        with patch.object(
            cog, "_get_http_session", new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image-pro",
                count=1,
            )

        cog.client.image.sample.assert_awaited_once()
        cog.client.image.sample_batch.assert_not_awaited()

    async def test_image_batch_calls_sample_batch(self, cog, mock_discord_context):
        """count>1 should call client.image.sample_batch with n=count."""
        with patch.object(
            cog, "_get_http_session", new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image-pro",
                count=3,
            )

        cog.client.image.sample.assert_not_awaited()
        cog.client.image.sample_batch.assert_awaited_once()
        call_kwargs = cog.client.image.sample_batch.call_args
        assert call_kwargs.kwargs["n"] == 3

    async def test_image_batch_sends_multiple_files(self, cog, mock_discord_context):
        """Batch generation should send multiple File objects."""
        with patch.object(
            cog, "_get_http_session", new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image",
                count=2,
            )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        files = call_kwargs["files"]
        assert len(files) == 2
        assert files[0].filename == "image_1.png"
        assert files[1].filename == "image_2.png"

    async def test_image_batch_cost_multiplied(self, cog, mock_discord_context):
        """Batch generation cost should be per-image cost × count."""
        from util import calculate_image_cost

        with patch.object(
            cog, "_get_http_session", new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image",
                count=2,
            )

        expected_cost = calculate_image_cost("grok-imagine-image") * 2
        # The daily cost tracker should reflect the multiplied cost
        from datetime import date
        key = (mock_discord_context.author.id, date.today().isoformat())
        assert abs(cog.daily_costs[key] - expected_cost) < 1e-9

    async def test_image_batch_rejects_editing_mode(
        self, cog, mock_discord_context, mock_attachment
    ):
        """count>1 with an attachment (editing mode) should return an error."""
        await cog.image.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Edit this",
            model="grok-imagine-image-pro",
            count=3,
            attachment=mock_attachment,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "not supported in Image Editing mode" in call_kwargs["embed"].description
        cog.client.image.sample.assert_not_awaited()
        cog.client.image.sample_batch.assert_not_awaited()


class TestHandleNewMessageInConversation:
    """Tests for the handle_new_message_in_conversation method."""

    @pytest.fixture
    def cog(self, mock_bot):
        return _make_cog(mock_bot)

    @pytest.fixture
    def conversation(self):
        from util import ChatCompletionParameters, Conversation

        starter = MagicMock()
        starter.id = 111222333
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=starter,
            channel_id=444555666,
            conversation_id=777888999,
        )
        return Conversation(
            params=params,
            previous_response_id="resp_prev",
            response_id_history=["resp_prev"],
            prompt_cache_key="test-cache-key",
        )

    @pytest.fixture
    def message(self, conversation):
        msg = MagicMock()
        msg.author = conversation.params.conversation_starter
        msg.author.id = 111222333
        msg.channel = MagicMock()
        msg.channel.id = 444555666
        msg.content = "Follow-up message"
        msg.attachments = []
        msg.reply = AsyncMock()
        msg.channel.typing = MagicMock()
        msg.channel.typing.return_value.__aenter__ = AsyncMock()
        msg.channel.typing.return_value.__aexit__ = AsyncMock()
        return msg

    async def test_follow_up_sends_response(self, cog, message, conversation):
        """A follow-up message should call the API and reply with embeds."""
        cog.conversations[777888999] = conversation
        cog.views[message.author] = MagicMock()

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_called_once()
        message.reply.assert_called()
        # Conversation state should be updated with new response ID
        assert conversation.previous_response_id == "resp_01XFDUDYJgAACzvnptvVoYEL"
        assert "resp_01XFDUDYJgAACzvnptvVoYEL" in conversation.response_id_history

    async def test_paused_conversation_ignored(self, cog, message, conversation):
        """Messages in a paused conversation should be silently ignored."""
        conversation.params.paused = True

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_not_called()
        message.reply.assert_not_called()

    async def test_wrong_author_ignored(self, cog, message, conversation):
        """Messages from a non-starter user should be silently ignored."""
        message.author = MagicMock()  # Different user

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_not_called()
        message.reply.assert_not_called()

    async def test_empty_content_returns_early(self, cog, message, conversation):
        """A message with no text and no attachments should return early."""
        message.content = ""
        message.attachments = []

        await cog.handle_new_message_in_conversation(message, conversation)

        cog._call_responses_api.assert_not_called()

    async def test_api_error_ends_conversation(self, cog, message, conversation):
        """An API error should end the conversation and reply with an error embed."""
        cog.conversations[777888999] = conversation
        cog._call_responses_api.side_effect = Exception("API failure")

        await cog.handle_new_message_in_conversation(message, conversation)

        message.reply.assert_called_once()
        embed = message.reply.call_args[1]["embed"]
        assert embed.title == "Error"
        assert 777888999 not in cog.conversations


class TestOnMessageRouting:
    """Tests for the on_message event listener routing."""

    @pytest.fixture
    def cog(self, mock_bot):
        return _make_cog(mock_bot)

    async def test_routes_to_correct_conversation(self, cog, mock_discord_message):
        """Message in the right channel from the right user should route to handler."""
        from util import ChatCompletionParameters, Conversation

        starter = mock_discord_message.author
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=starter,
            channel_id=mock_discord_message.channel.id,
            conversation_id=111,
        )
        cog.conversations[111] = Conversation(params=params, prompt_cache_key="k")
        cog.handle_new_message_in_conversation = AsyncMock()

        await cog.on_message(mock_discord_message)

        cog.handle_new_message_in_conversation.assert_awaited_once()

    async def test_wrong_channel_skipped(self, cog, mock_discord_message):
        """Message in a different channel should not route."""
        from util import ChatCompletionParameters, Conversation

        starter = mock_discord_message.author
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=starter,
            channel_id=999999,  # Different channel
            conversation_id=111,
        )
        cog.conversations[111] = Conversation(params=params, prompt_cache_key="k")
        cog.handle_new_message_in_conversation = AsyncMock()

        await cog.on_message(mock_discord_message)

        cog.handle_new_message_in_conversation.assert_not_awaited()

    async def test_wrong_author_skipped(self, cog, mock_discord_message):
        """Message from a different user should not route."""
        from util import ChatCompletionParameters, Conversation

        other_user = MagicMock()
        other_user.id = 999
        params = ChatCompletionParameters(
            model="grok-3",
            conversation_starter=other_user,
            channel_id=mock_discord_message.channel.id,
            conversation_id=111,
        )
        cog.conversations[111] = Conversation(params=params, prompt_cache_key="k")
        cog.handle_new_message_in_conversation = AsyncMock()

        await cog.on_message(mock_discord_message)

        cog.handle_new_message_in_conversation.assert_not_awaited()


class TestAppendSourcesEmbed:
    """Tests for the append_sources_embed helper."""

    def test_empty_citations_no_embed(self):
        from xai_api import append_sources_embed

        embeds = []
        append_sources_embed(embeds, [])
        assert len(embeds) == 0

    def test_web_citations_grouped(self):
        from xai_api import append_sources_embed

        citations = [
            {"url": "https://example.com/a", "source": "web"},
            {"url": "https://example.com/b", "source": "web"},
        ]
        embeds = []
        append_sources_embed(embeds, citations)
        assert len(embeds) == 1
        assert embeds[0].title == "Sources"
        assert "example.com" in embeds[0].description

    def test_mixed_sources_have_headings(self):
        from xai_api import append_sources_embed

        citations = [
            {"url": "https://example.com/a", "source": "web"},
            {"url": "https://x.com/i/status/123", "source": "x"},
        ]
        embeds = []
        append_sources_embed(embeds, citations)
        assert "**Web**" in embeds[0].description
        assert "**X Posts**" in embeds[0].description

    def test_single_source_type_no_heading(self):
        from xai_api import append_sources_embed

        citations = [
            {"url": "https://x.com/i/status/1", "source": "x"},
            {"url": "https://x.com/i/status/2", "source": "x"},
        ]
        embeds = []
        append_sources_embed(embeds, citations)
        assert "**X Posts**" not in embeds[0].description

    def test_skips_when_at_embed_limit(self):
        from xai_api import append_sources_embed

        embeds = [MagicMock() for _ in range(10)]
        citations = [{"url": "https://example.com", "source": "web"}]
        append_sources_embed(embeds, citations)
        assert len(embeds) == 10  # No embed added


class TestChatMutualExclusionParams:
    """Tests for mutual exclusion parameter validation."""

    @pytest.fixture
    def cog(self, mock_bot):
        return _make_cog(mock_bot)

    async def test_rejects_both_x_search_handles(self, cog, mock_discord_context):
        """Setting both allowed and excluded X handles should error."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="test",
            x_search=True,
            x_search_allowed_handles="a",
            x_search_excluded_handles="b",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "Cannot use both" in call_kwargs["embed"].description

    async def test_rejects_both_web_search_domains(self, cog, mock_discord_context):
        """Setting both allowed and excluded web domains should error."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="test",
            web_search=True,
            web_search_allowed_domains="a.com",
            web_search_excluded_domains="b.com",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "Cannot use both" in call_kwargs["embed"].description

    async def test_rejects_too_many_x_handles(self, cog, mock_discord_context):
        """More than 10 X handles should be rejected."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        handles = ",".join(f"user{i}" for i in range(11))
        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="test",
            x_search=True,
            x_search_allowed_handles=handles,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "maximum of 10" in call_kwargs["embed"].description

    async def test_rejects_too_many_web_domains(self, cog, mock_discord_context):
        """More than 5 web domains should be rejected."""
        mock_discord_context.channel.typing = MagicMock()
        mock_discord_context.channel.typing.return_value.__aenter__ = AsyncMock()
        mock_discord_context.channel.typing.return_value.__aexit__ = AsyncMock()

        domains = ",".join(f"d{i}.com" for i in range(6))
        await cog.chat.callback(
            cog,
            ctx=mock_discord_context,
            prompt="test",
            web_search=True,
            web_search_allowed_domains=domains,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "maximum of 5" in call_kwargs["embed"].description


class TestVideoCommand:
    """Integration tests for the /grok video command."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client):
        cog = _make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    @staticmethod
    def _mock_http_session():
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"fake video bytes")

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm
        return mock_session

    async def test_video_success(self, cog, mock_discord_context):
        """Successful text-to-video should send a video file."""
        with patch.object(
            cog, "_get_http_session", new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.video.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A sunset",
            )

        cog.client.video.generate.assert_awaited_once()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["file"].filename == "video.mp4"

    async def test_video_with_attachment(self, cog, mock_discord_context, mock_attachment):
        """Image-to-video should pass image_url to the SDK."""
        with patch.object(
            cog, "_get_http_session", new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.video.callback(
                cog,
                ctx=mock_discord_context,
                prompt="Animate this",
                attachment=mock_attachment,
            )

        gen_kwargs = cog.client.video.generate.call_args[1]
        assert gen_kwargs["image_url"] == str(mock_attachment.url)

    async def test_video_api_error(self, cog, mock_discord_context):
        """API errors should display an error embed."""
        cog.client.video.generate.side_effect = Exception("Video gen failed")

        await cog.video.callback(
            cog,
            ctx=mock_discord_context,
            prompt="A sunset",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embed"].title == "Error"

    async def test_video_no_url_returns_error(self, cog, mock_discord_context):
        """No video URL from API should display an error."""
        cog.client.video.generate.return_value.url = None

        await cog.video.callback(
            cog,
            ctx=mock_discord_context,
            prompt="A sunset",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embed"].title == "Error"


class TestResolveSelectedToolsUtil:
    """Tests for the resolve_selected_tools function in util.py."""

    def test_basic_resolution(self):
        from util import resolve_selected_tools

        tools, error = resolve_selected_tools(
            ["web_search", "x_search", "code_execution"],
            collection_ids=[],
        )
        assert error is None
        assert len(tools) == 3

    def test_collections_requires_ids(self):
        from util import resolve_selected_tools

        tools, error = resolve_selected_tools(
            ["collections_search"],
            collection_ids=[],
        )
        assert tools == []
        assert "XAI_COLLECTION_IDS" in error

    def test_collections_with_ids(self):
        from util import resolve_selected_tools

        tools, error = resolve_selected_tools(
            ["collections_search"],
            collection_ids=["col_1"],
        )
        assert error is None
        assert tools[0]["type"] == "file_search"
        assert tools[0]["vector_store_ids"] == ["col_1"]
