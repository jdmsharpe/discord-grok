import pytest

from discord_grok.cogs.grok.tooling import (
    CHUNK_TEXT_SIZE,
    IMAGE_PRICING,
    MODEL_PRICING,
    TOOL_BUILDERS,
    TOOL_CODE_EXECUTION,
    TOOL_COLLECTIONS_SEARCH,
    TOOL_INVOCATION_PRICING,
    TOOL_REMOTE_MCP,
    TOOL_WEB_SEARCH,
    TOOL_X_SEARCH,
    TTS_PRICING_PER_MILLION_CHARS,
    VIDEO_PRICING,
    ChatCompletionParameters,
    Conversation,
    McpServerConfig,
    calculate_cost,
    calculate_image_cost,
    calculate_tool_cost,
    calculate_tts_cost,
    calculate_video_cost,
    chunk_text,
    format_xai_error,
    resolve_tool_name,
    truncate_text,
    validate_mcp_server_input,
)


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_short_text_single_chunk(self):
        """Short text should return a single chunk."""
        text = "Hello, world!"
        result = chunk_text(text)
        assert result == ["Hello, world!"]

    def test_exact_chunk_size(self):
        """Text exactly at chunk size should return one chunk."""
        text = "a" * CHUNK_TEXT_SIZE
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_text_splits_into_multiple_chunks(self):
        """Text longer than chunk size should split into multiple chunks."""
        text = "a" * (CHUNK_TEXT_SIZE * 2 + 100)
        result = chunk_text(text)
        assert len(result) == 3
        assert len(result[0]) == CHUNK_TEXT_SIZE
        assert len(result[1]) == CHUNK_TEXT_SIZE
        assert len(result[2]) == 100

    def test_custom_chunk_size(self):
        """Custom chunk size should be respected."""
        text = "Hello, world! This is a test."
        result = chunk_text(text, chunk_size=10)
        assert len(result) == 3
        assert result[0] == "Hello, wor"
        assert result[1] == "ld! This i"
        assert result[2] == "s a test."

    def test_empty_string(self):
        """Empty string should return empty list."""
        result = chunk_text("")
        assert result == []


class TestTruncateText:
    """Tests for the truncate_text function."""

    def test_short_text_unchanged(self):
        """Text shorter than max_length should be unchanged."""
        text = "Hello"
        result = truncate_text(text, 10)
        assert result == "Hello"

    def test_exact_length_unchanged(self):
        """Text at exact max_length should be unchanged."""
        text = "Hello"
        result = truncate_text(text, 5)
        assert result == "Hello"

    def test_long_text_truncated(self):
        """Text longer than max_length should be truncated with suffix."""
        text = "Hello, world!"
        result = truncate_text(text, 8)
        assert result == "Hello, w..."

    def test_custom_suffix(self):
        """Custom suffix should be used."""
        text = "Hello, world!"
        result = truncate_text(text, 8, suffix="[cut]")
        assert result == "Hello, w[cut]"

    def test_none_returns_none(self):
        """None input should return None."""
        result = truncate_text(None, 10)
        assert result is None


class TestFormatXAIError:
    """Tests for the format_xai_error function."""

    def test_basic_exception(self):
        """Basic exception should format correctly."""
        error = Exception("Something went wrong")
        result = format_xai_error(error)
        assert "Something went wrong" in result

    def test_exception_with_status_code(self):
        """Exception with status_code attribute should include it."""
        error = Exception("API error")
        error.status_code = 429
        result = format_xai_error(error)
        assert "API error" in result
        assert "Status: 429" in result

    def test_exception_with_message_attribute(self):
        """Exception with message attribute should use it."""
        error = Exception()
        error.message = "Custom message"
        result = format_xai_error(error)
        assert "Custom message" in result

    def test_exception_with_code_attribute(self):
        """Exception with code attribute (gRPC style) should include it."""
        error = Exception("gRPC error")
        error.code = 14
        result = format_xai_error(error)
        assert "gRPC error" in result
        assert "Status: 14" in result

    def test_grpc_aio_rpc_error_is_humanized(self):
        """A real grpc.aio.AioRpcError (code()/details() are methods, no
        .message, verbose __str__) must be humanized to its details + status,
        not leak a bound-method repr or duplicate the verbose repr."""

        class _Status:
            name = "INVALID_ARGUMENT"

        class _FakeAioRpcError(Exception):
            def code(self):
                return _Status()

            def details(self):
                return "Text-to-video is not supported for this model."

            def __str__(self):
                return (
                    "<_FakeAioRpcError of RPC that terminated with:\n"
                    "\tstatus = StatusCode.INVALID_ARGUMENT\n"
                    '\tdetails = "Text-to-video is not supported for this model."\n>'
                )

        result = format_xai_error(_FakeAioRpcError())
        assert "Text-to-video is not supported for this model." in result
        assert "Status: INVALID_ARGUMENT" in result
        assert "bound method" not in result
        assert "RPC that terminated with" not in result


class TestChatCompletionParameters:
    """Tests for the ChatCompletionParameters dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        params = ChatCompletionParameters(model="grok-4.3")
        assert params.model == "grok-4.3"
        assert params.temperature is None
        assert params.top_p is None
        assert params.max_tokens is None
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.reasoning_effort is None
        assert params.agent_count is None
        assert params.tools == []
        assert params.mcp_servers == []
        assert params.x_search_kwargs == {}
        assert params.web_search_kwargs == {}
        assert params.paused is False
        assert params.conversation_id is None
        assert params.channel_id is None

    def test_all_params_set(self):
        """All parameters should be stored correctly."""
        params = ChatCompletionParameters(
            model="grok-4.3",
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            reasoning_effort="high",
            tools=[
                TOOL_BUILDERS[TOOL_WEB_SEARCH](),
                TOOL_BUILDERS[TOOL_X_SEARCH](),
                TOOL_BUILDERS[TOOL_CODE_EXECUTION](),
            ],
            mcp_servers=[
                McpServerConfig(
                    server_url="https://mcp.example.com/sse",
                    server_label="mcp.example.com",
                    allowed_tool_names=["search"],
                )
            ],
        )
        assert params.model == "grok-4.3"
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 2048
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.3
        assert params.reasoning_effort == "high"
        assert len(params.tools) == 3
        assert params.mcp_servers[0].server_url == "https://mcp.example.com/sse"
        assert resolve_tool_name(params.tools[0]) == TOOL_WEB_SEARCH
        assert resolve_tool_name(params.tools[1]) == TOOL_X_SEARCH
        assert resolve_tool_name(params.tools[2]) == TOOL_CODE_EXECUTION

    def test_default_tools_isolated(self):
        """Default tools list should not be shared across instances."""
        params_one = ChatCompletionParameters(model="grok-4.3")
        params_one.tools.append(TOOL_BUILDERS[TOOL_WEB_SEARCH]())

        params_two = ChatCompletionParameters(model="grok-4.3")
        assert params_two.tools == []
        assert params_one.tools is not params_two.tools

    def test_default_mcp_servers_isolated(self):
        params_one = ChatCompletionParameters(model="grok-4.3")
        params_one.mcp_servers.append(
            McpServerConfig(
                server_url="https://mcp.example.com/sse",
                server_label="mcp.example.com",
            )
        )

        params_two = ChatCompletionParameters(model="grok-4.3")
        assert params_two.mcp_servers == []
        assert params_one.mcp_servers is not params_two.mcp_servers


class TestModelLists:
    """Tests for the model list constants."""

    def test_grok_models_contains_new_4_20_models(self):
        """GROK_MODELS should include all grok-4.20 GA models."""
        from discord_grok.cogs.grok.tooling import GROK_MODELS

        assert "grok-4.20-multi-agent" in GROK_MODELS
        assert "grok-4.20" in GROK_MODELS
        assert "grok-4.20-non-reasoning" in GROK_MODELS

    def test_grok_models_no_deprecated(self):
        """GROK_MODELS should not contain deprecated grok-2 models."""
        from discord_grok.cogs.grok.tooling import GROK_MODELS

        for model in GROK_MODELS:
            assert not model.startswith("grok-2"), f"Deprecated model found: {model}"

    def test_grok_image_models_no_deprecated(self):
        """GROK_IMAGE_MODELS should not contain deprecated grok-2 models."""
        from discord_grok.cogs.grok.tooling import GROK_IMAGE_MODELS

        for model in GROK_IMAGE_MODELS:
            assert not model.startswith("grok-2"), f"Deprecated model found: {model}"


class TestReasoningConstants:
    """Tests for reasoning model constants."""

    def test_penalty_supported_models_are_non_reasoning(self):
        from discord_grok.cogs.grok.tooling import PENALTY_SUPPORTED_MODELS

        for model in PENALTY_SUPPORTED_MODELS:
            assert "non-reasoning" in model

    def test_reasoning_models_excluded_from_penalty_support(self):
        from discord_grok.cogs.grok.tooling import PENALTY_SUPPORTED_MODELS

        assert "grok-4.3" not in PENALTY_SUPPORTED_MODELS
        assert "grok-4.20" not in PENALTY_SUPPORTED_MODELS
        assert "grok-4.5" not in PENALTY_SUPPORTED_MODELS

    def test_reasoning_effort_models(self):
        from discord_grok.cogs.grok.tooling import REASONING_EFFORT_MODELS

        assert {"grok-4.3", "grok-4.5"} == REASONING_EFFORT_MODELS

    def test_model_reasoning_efforts_per_model(self):
        from discord_grok.cogs.grok.tooling import MODEL_REASONING_EFFORTS

        assert MODEL_REASONING_EFFORTS["grok-4.3"] == frozenset({"none", "low", "medium", "high"})
        assert "grok-4.20" not in MODEL_REASONING_EFFORTS

    def test_grok_4_5_reasoning_cannot_be_disabled(self):
        """xAI rejects reasoning_effort="none" on grok-4.5; exposing it would 400 live."""
        from discord_grok.cogs.grok.tooling import MODEL_REASONING_EFFORTS

        assert MODEL_REASONING_EFFORTS["grok-4.5"] == frozenset({"low", "medium", "high"})
        assert "none" not in MODEL_REASONING_EFFORTS["grok-4.5"]

    def test_multi_agent_models(self):
        from discord_grok.cogs.grok.tooling import MULTI_AGENT_MODELS

        assert "grok-4.20-multi-agent" in MULTI_AGENT_MODELS
        for model in MULTI_AGENT_MODELS:
            assert "multi-agent" in model


class TestTTSConstants:
    """Tests for TTS-related constants."""

    def test_tts_voices_contains_all_voices(self):
        from discord_grok.cogs.grok.tooling import TTS_VOICES

        assert TTS_VOICES == ["eve", "ara", "rex", "sal", "leo"]

    def test_tts_voices_has_five_entries(self):
        from discord_grok.cogs.grok.tooling import TTS_VOICES

        assert len(TTS_VOICES) == 5


class TestToolHelpers:
    """Tests for tool helper constants and functions."""

    def test_resolve_tool_name_for_builtin_tools(self):
        assert resolve_tool_name(TOOL_BUILDERS[TOOL_WEB_SEARCH]()) == TOOL_WEB_SEARCH
        assert resolve_tool_name(TOOL_BUILDERS[TOOL_X_SEARCH]()) == TOOL_X_SEARCH
        assert resolve_tool_name(TOOL_BUILDERS[TOOL_CODE_EXECUTION]()) == TOOL_CODE_EXECUTION

    def test_resolve_tool_name_for_collections_search(self):
        tool = {"type": "file_search", "vector_store_ids": ["collection_123"]}
        assert resolve_tool_name(tool) == TOOL_COLLECTIONS_SEARCH

    def test_resolve_tool_name_for_mcp(self):
        tool = {
            "type": "mcp",
            "server_url": "https://mcp.example.com/sse",
            "server_label": "mcp.example.com",
        }
        assert resolve_tool_name(tool) == TOOL_REMOTE_MCP

    def test_resolve_tool_name_unknown(self):
        assert resolve_tool_name(object()) is None
        assert resolve_tool_name({"type": "unknown_tool"}) is None

    def test_validate_mcp_server_input_success(self):
        config, error = validate_mcp_server_input(
            "https://www.example.com/mcp",
            "search, run, search",
        )

        assert error is None
        assert config is not None
        assert config.server_url == "https://www.example.com/mcp"
        assert config.server_label == "example.com"
        assert config.allowed_tool_names == ["search", "run"]

    def test_validate_mcp_server_input_requires_https(self):
        config, error = validate_mcp_server_input("http://example.com/mcp")

        assert config is None
        assert "HTTPS" in error

    def test_validate_mcp_server_input_rejects_too_many_tool_names(self):
        allowed = ",".join(f"tool_{index}" for index in range(21))

        config, error = validate_mcp_server_input("https://example.com/mcp", allowed)

        assert config is None
        assert "maximum of 20" in error


class TestPricing:
    """Tests for pricing constants and cost calculation functions."""

    def test_model_pricing_covers_all_grok_models(self):
        """Every model in GROK_MODELS should have a pricing entry."""
        from discord_grok.cogs.grok.tooling import GROK_MODELS

        for model in GROK_MODELS:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"

    def test_image_pricing_covers_all_image_models(self):
        """Every model in GROK_IMAGE_MODELS should have a pricing entry."""
        from discord_grok.cogs.grok.tooling import GROK_IMAGE_MODELS

        for model in GROK_IMAGE_MODELS:
            assert model in IMAGE_PRICING, f"Missing pricing for {model}"

    def test_calculate_cost_known_model(self):
        """Cost should use the model's pricing rates."""
        # grok-4.20 (flagship): $1.25/M in, $2.50/M out; a 100k prompt stays
        # below the 200k long-context threshold.
        cost = calculate_cost("grok-4.20", 100_000, 1_000_000)
        assert cost == pytest.approx(0.125 + 2.50)

    def test_calculate_cost_unknown_model_uses_default(self):
        """Unknown models should fall back to the default model's pricing
        (grok-4.5 since the 2026-07-09 promotion: $2.00/M in, $6.00/M out).
        The fallback is flat-only: without verified tier data no long-context
        threshold applies, even though this 1M prompt would cross grok-4.5's."""
        cost = calculate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == 2.00 + 6.00

    def test_calculate_cost_with_reasoning_tokens(self):
        """Reasoning tokens should be billed at the output rate."""
        # grok-4.20 (flagship): $1.25/M in, $2.50/M out
        cost = calculate_cost("grok-4.20", 100_000, 500_000, reasoning_tokens=500_000)
        # 100k in * $1.25/M + (500k out + 500k reasoning) * $2.50/M
        assert cost == pytest.approx(0.125 + 2.50)

    def test_calculate_cost_zero_tokens(self):
        """Zero tokens should return zero cost."""
        assert calculate_cost("grok-4.20", 0, 0) == 0.0

    def test_calculate_image_cost_known_model(self):
        assert calculate_image_cost("grok-imagine-image") == 0.02
        assert calculate_image_cost("grok-imagine-image-pro") == 0.05
        assert calculate_image_cost("grok-imagine-image-quality") == 0.05

    def test_calculate_image_cost_unknown_model(self):
        assert calculate_image_cost("unknown") == 0.05

    def test_calculate_cost_with_cached_tokens(self):
        """Cached tokens should be billed at the discounted rate."""
        # grok-4.20 (flagship): $1.25/M in, $0.20/M cached, $2.50/M out
        # 100k input with 50k cached: 50k * $1.25/M + 50k * $0.20/M + 0 out
        cost = calculate_cost("grok-4.20", 100_000, 0, cached_tokens=50_000)
        assert cost == pytest.approx(0.0625 + 0.01)

    def test_calculate_cost_all_cached(self):
        """If all input tokens are cached, only cached rate applies."""
        # grok-4.20 (flagship): $0.20/M cached
        cost = calculate_cost("grok-4.20", 100_000, 0, cached_tokens=100_000)
        assert cost == pytest.approx(0.02)

    def test_calculate_cost_grok_4_5_cached_rate_is_not_premium(self):
        """grok-4.5 caches at $0.50/M, not the $0.20/M `premium` uses. Reusing `premium`
        for it would under-bill every cached read by 2.5x and this test would catch it."""
        cost = calculate_cost("grok-4.5", 100_000, 0, cached_tokens=100_000)
        assert cost == pytest.approx(0.05)

    def test_calculate_cost_below_long_context_threshold(self):
        """A 199,999-token prompt is one token short of the long-context tier."""
        # grok-4.20 (flagship): standard $1.25/M in, $2.50/M out
        cost = calculate_cost("grok-4.20", 199_999, 1_000_000)
        assert cost == pytest.approx((199_999 / 1_000_000) * 1.25 + 2.50)

    def test_calculate_cost_at_long_context_threshold(self):
        """The tier boundary is inclusive: per docs.x.ai a prompt that *reaches*
        200k tokens bills ALL tokens in the request at the higher rate — every
        input token plus the output, not just the overflow past 200k."""
        # grok-4.20 long tier: $2.50/M in, $5.00/M out
        cost = calculate_cost("grok-4.20", 200_000, 1_000_000)
        assert cost == pytest.approx((200_000 / 1_000_000) * 2.50 + 5.00)

    def test_calculate_cost_above_long_context_threshold(self):
        """Above the threshold every flagship rate doubles, and reasoning tokens
        bill at the long-tier output rate."""
        # 1M in * $2.50/M + (500k out + 500k reasoning) * $5.00/M
        cost = calculate_cost("grok-4.20", 1_000_000, 500_000, reasoning_tokens=500_000)
        assert cost == pytest.approx(2.50 + 5.00)

    def test_calculate_cost_cached_tokens_long_tier(self):
        """Cached input doubles in the long tier too: $0.20/M → $0.40/M flagship."""
        cost = calculate_cost("grok-4.20", 1_000_000, 0, cached_tokens=1_000_000)
        assert cost == pytest.approx(0.40)

    def test_calculate_cost_grok_4_5_long_tier(self):
        """grok-4.5's long tier: $4.00/M in, $1.00/M cached, $12.00/M out."""
        # 200k non-cached * $4.00/M + 200k cached * $1.00/M + 100k out * $12.00/M
        cost = calculate_cost("grok-4.5", 400_000, 100_000, cached_tokens=200_000)
        assert cost == pytest.approx(0.80 + 0.20 + 1.20)

    def test_calculate_cost_model_without_tier_stays_flat(self, monkeypatch):
        """Models whose pricing class has no long_context block bill flat at any
        prompt size. Every catalog model is tiered today, so simulate a future
        flat-class model by dropping the tier entry."""
        from discord_grok.cogs.grok import tooling

        monkeypatch.delitem(tooling.MODEL_LONG_CONTEXT_PRICING, "grok-4.20")
        cost = calculate_cost("grok-4.20", 1_000_000, 1_000_000)
        assert cost == 1.25 + 2.50

    def test_model_pricing_has_three_values(self):
        """Each MODEL_PRICING entry should be a 3-tuple (input, cached, output)."""
        for model, prices in MODEL_PRICING.items():
            assert len(prices) == 3, f"{model} pricing should have 3 values"

    def test_model_long_context_pricing_has_four_values(self):
        """Each MODEL_LONG_CONTEXT_PRICING entry should be a 4-tuple
        (threshold, input, cached, output)."""
        from discord_grok.cogs.grok.tooling import MODEL_LONG_CONTEXT_PRICING

        for model, tier in MODEL_LONG_CONTEXT_PRICING.items():
            assert len(tier) == 4, f"{model} long-context tier should have 4 values"

    def test_calculate_tool_cost_known_tools(self):
        """Tool invocations should be billed at per-1k rates."""
        # 1000 web searches at $5/1k = $5
        cost = calculate_tool_cost({"SERVER_SIDE_TOOL_WEB_SEARCH": 1000})
        assert cost == pytest.approx(5.00)

    def test_calculate_tool_cost_fractional(self):
        """Fewer than 1k invocations should cost proportionally."""
        # 1 web search at $5/1k = $0.005
        cost = calculate_tool_cost({"SERVER_SIDE_TOOL_WEB_SEARCH": 1})
        assert cost == pytest.approx(0.005)

    def test_calculate_tool_cost_multiple_tools(self):
        """Multiple tool types should sum their costs."""
        cost = calculate_tool_cost(
            {
                "SERVER_SIDE_TOOL_WEB_SEARCH": 1000,
                "SERVER_SIDE_TOOL_COLLECTIONS_SEARCH": 1000,
            }
        )
        assert cost == pytest.approx(5.00 + 2.50)

    def test_calculate_tool_cost_unknown_tool(self):
        """Unknown tool keys should contribute zero cost."""
        cost = calculate_tool_cost({"SERVER_SIDE_TOOL_UNKNOWN": 100})
        assert cost == 0.0

    def test_calculate_tool_cost_empty(self):
        """Empty tool usage should return zero."""
        assert calculate_tool_cost({}) == 0.0

    def test_tool_invocation_pricing_keys(self):
        """TOOL_INVOCATION_PRICING should cover known server-side tools."""
        assert "SERVER_SIDE_TOOL_WEB_SEARCH" in TOOL_INVOCATION_PRICING
        assert "SERVER_SIDE_TOOL_X_SEARCH" in TOOL_INVOCATION_PRICING
        assert "SERVER_SIDE_TOOL_CODE_EXECUTION" in TOOL_INVOCATION_PRICING
        assert "SERVER_SIDE_TOOL_COLLECTIONS_SEARCH" in TOOL_INVOCATION_PRICING
        assert "SERVER_SIDE_TOOL_ATTACHMENT_SEARCH" in TOOL_INVOCATION_PRICING

    def test_calculate_tts_cost(self):
        """TTS cost should be based on character count."""
        # 1M chars at $15.00/M = $15.00
        cost = calculate_tts_cost(1_000_000)
        assert cost == pytest.approx(TTS_PRICING_PER_MILLION_CHARS)

    def test_calculate_tts_cost_small(self):
        """Small TTS should be proportional."""
        # 100 chars at $15.00/M
        cost = calculate_tts_cost(100)
        assert cost == pytest.approx(100 / 1_000_000 * TTS_PRICING_PER_MILLION_CHARS)

    def test_calculate_tts_cost_zero(self):
        """Zero characters should return zero cost."""
        assert calculate_tts_cost(0) == 0.0

    def test_calculate_video_cost(self):
        # Bare call uses the default video model (grok-imagine-video-1.5-preview).
        assert calculate_video_cost(5) == 5 * VIDEO_PRICING["grok-imagine-video-1.5-preview"]
        assert calculate_video_cost(5, "grok-imagine-video") == (
            5 * VIDEO_PRICING["grok-imagine-video"]
        )
        assert calculate_video_cost(0) == 0.0


class TestConversation:
    """Tests for the Conversation dataclass."""

    def test_conversation_creation(self):
        """Conversation should store params and response ID state."""
        params = ChatCompletionParameters(model="grok-4.3")
        conv = Conversation(params=params)

        assert conv.params == params
        assert conv.previous_response_id is None
        assert conv.response_id_history == []
        assert conv.file_ids == []
        assert conv.grok_conv_id is None

    def test_conversation_with_response_id(self):
        """Conversation should store response ID and history."""
        params = ChatCompletionParameters(model="grok-4.3")
        conv = Conversation(
            params=params,
            previous_response_id="resp_123",
            response_id_history=["resp_123"],
            file_ids=["file-1", "file-2"],
            grok_conv_id="conv_123",
        )
        assert conv.previous_response_id == "resp_123"
        assert conv.response_id_history == ["resp_123"]
        assert conv.file_ids == ["file-1", "file-2"]
        assert conv.grok_conv_id == "conv_123"

    def test_default_file_ids_isolated(self):
        """Default file_ids list should not be shared across instances."""
        params = ChatCompletionParameters(model="grok-4.3")
        conv_one = Conversation(params=params)
        conv_one.file_ids.append("file-1")

        conv_two = Conversation(params=params)
        assert conv_two.file_ids == []
        assert conv_one.file_ids is not conv_two.file_ids

    def test_default_response_id_history_isolated(self):
        """Default response_id_history should not be shared across instances."""
        params = ChatCompletionParameters(model="grok-4.3")
        conv_one = Conversation(params=params)
        conv_one.response_id_history.append("resp_1")

        conv_two = Conversation(params=params)
        assert conv_two.response_id_history == []
        assert conv_one.response_id_history is not conv_two.response_id_history
