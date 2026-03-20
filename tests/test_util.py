import pytest

from src.util import (
    CHUNK_TEXT_SIZE,
    IMAGE_PRICING,
    MODEL_PRICING,
    TOOL_INVOCATION_PRICING,
    TTS_PRICING_PER_MILLION_CHARS,
    VIDEO_PRICING_PER_SECOND,
    ChatCompletionParameters,
    Conversation,
    TOOL_BUILDERS,
    TOOL_CODE_EXECUTION,
    TOOL_COLLECTIONS_SEARCH,
    TOOL_WEB_SEARCH,
    TOOL_X_SEARCH,
    calculate_cost,
    calculate_image_cost,
    calculate_tool_cost,
    calculate_tts_cost,
    calculate_video_cost,
    chunk_text,
    format_xai_error,
    resolve_tool_name,
    truncate_text,
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
        setattr(error, "status_code", 429)
        result = format_xai_error(error)
        assert "API error" in result
        assert "Status: 429" in result

    def test_exception_with_message_attribute(self):
        """Exception with message attribute should use it."""
        error = Exception()
        setattr(error, "message", "Custom message")
        result = format_xai_error(error)
        assert "Custom message" in result

    def test_exception_with_code_attribute(self):
        """Exception with code attribute (gRPC style) should include it."""
        error = Exception("gRPC error")
        setattr(error, "code", 14)
        result = format_xai_error(error)
        assert "gRPC error" in result
        assert "Status: 14" in result


class TestChatCompletionParameters:
    """Tests for the ChatCompletionParameters dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        params = ChatCompletionParameters(model="grok-3")
        assert params.model == "grok-3"
        assert params.temperature is None
        assert params.top_p is None
        assert params.max_tokens is None
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.reasoning_effort is None
        assert params.agent_count is None
        assert params.tools == []
        assert params.x_search_kwargs == {}
        assert params.web_search_kwargs == {}
        assert params.paused is False
        assert params.conversation_id is None
        assert params.channel_id is None

    def test_all_params_set(self):
        """All parameters should be stored correctly."""
        params = ChatCompletionParameters(
            model="grok-4.20",
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
        )
        assert params.model == "grok-4.20"
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 2048
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.3
        assert params.reasoning_effort == "high"
        assert len(params.tools) == 3
        assert resolve_tool_name(params.tools[0]) == TOOL_WEB_SEARCH
        assert resolve_tool_name(params.tools[1]) == TOOL_X_SEARCH
        assert resolve_tool_name(params.tools[2]) == TOOL_CODE_EXECUTION

    def test_default_tools_isolated(self):
        """Default tools list should not be shared across instances."""
        params_one = ChatCompletionParameters(model="grok-3")
        params_one.tools.append(TOOL_BUILDERS[TOOL_WEB_SEARCH]())

        params_two = ChatCompletionParameters(model="grok-3")
        assert params_two.tools == []
        assert params_one.tools is not params_two.tools


class TestModelLists:
    """Tests for the model list constants."""

    def test_grok_models_contains_new_4_20_models(self):
        """GROK_MODELS should include all grok-4.20 GA models."""
        from src.util import GROK_MODELS

        assert "grok-4.20-multi-agent" in GROK_MODELS
        assert "grok-4.20" in GROK_MODELS
        assert "grok-4.20-non-reasoning" in GROK_MODELS

    def test_grok_models_no_deprecated(self):
        """GROK_MODELS should not contain deprecated grok-2 models."""
        from src.util import GROK_MODELS

        for model in GROK_MODELS:
            assert not model.startswith("grok-2"), f"Deprecated model found: {model}"

    def test_grok_image_models_no_deprecated(self):
        """GROK_IMAGE_MODELS should not contain deprecated grok-2 models."""
        from src.util import GROK_IMAGE_MODELS

        for model in GROK_IMAGE_MODELS:
            assert not model.startswith("grok-2"), f"Deprecated model found: {model}"


class TestReasoningConstants:
    """Tests for reasoning model constants."""

    def test_penalty_supported_models_are_non_reasoning(self):
        from src.util import PENALTY_SUPPORTED_MODELS

        for model in PENALTY_SUPPORTED_MODELS:
            assert "non-reasoning" in model

    def test_reasoning_models_excluded_from_penalty_support(self):
        from src.util import PENALTY_SUPPORTED_MODELS

        assert "grok-3-mini" not in PENALTY_SUPPORTED_MODELS
        assert "grok-3" not in PENALTY_SUPPORTED_MODELS
        assert "grok-4-0709" not in PENALTY_SUPPORTED_MODELS

    def test_reasoning_effort_models(self):
        from src.util import REASONING_EFFORT_MODELS

        assert REASONING_EFFORT_MODELS == {"grok-3-mini"}

    def test_multi_agent_models(self):
        from src.util import MULTI_AGENT_MODELS

        assert "grok-4.20-multi-agent" in MULTI_AGENT_MODELS
        for model in MULTI_AGENT_MODELS:
            assert "multi-agent" in model


class TestTTSConstants:
    """Tests for TTS-related constants."""

    def test_tts_voices_contains_all_voices(self):
        from src.util import TTS_VOICES

        assert TTS_VOICES == ["eve", "ara", "rex", "sal", "leo"]

    def test_tts_voices_has_five_entries(self):
        from src.util import TTS_VOICES

        assert len(TTS_VOICES) == 5


class TestToolHelpers:
    """Tests for tool helper constants and functions."""

    def test_resolve_tool_name_for_builtin_tools(self):
        assert resolve_tool_name(TOOL_BUILDERS[TOOL_WEB_SEARCH]()) == TOOL_WEB_SEARCH
        assert resolve_tool_name(TOOL_BUILDERS[TOOL_X_SEARCH]()) == TOOL_X_SEARCH
        assert (
            resolve_tool_name(TOOL_BUILDERS[TOOL_CODE_EXECUTION]())
            == TOOL_CODE_EXECUTION
        )

    def test_resolve_tool_name_for_collections_search(self):
        tool = {"type": "file_search", "vector_store_ids": ["collection_123"]}
        assert resolve_tool_name(tool) == TOOL_COLLECTIONS_SEARCH

    def test_resolve_tool_name_unknown(self):
        assert resolve_tool_name(object()) is None
        assert resolve_tool_name({"type": "unknown_tool"}) is None


class TestPricing:
    """Tests for pricing constants and cost calculation functions."""

    def test_model_pricing_covers_all_grok_models(self):
        """Every model in GROK_MODELS should have a pricing entry."""
        from src.util import GROK_MODELS

        for model in GROK_MODELS:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"

    def test_image_pricing_covers_all_image_models(self):
        """Every model in GROK_IMAGE_MODELS should have a pricing entry."""
        from src.util import GROK_IMAGE_MODELS

        for model in GROK_IMAGE_MODELS:
            assert model in IMAGE_PRICING, f"Missing pricing for {model}"

    def test_calculate_cost_known_model(self):
        """Cost should use the model's pricing rates."""
        cost = calculate_cost("grok-3", 1_000_000, 1_000_000)
        assert cost == 3.00 + 15.00

    def test_calculate_cost_unknown_model_uses_default(self):
        """Unknown models should fall back to the default pricing."""
        cost = calculate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == 2.00 + 6.00

    def test_calculate_cost_with_reasoning_tokens(self):
        """Reasoning tokens should be billed at the output rate."""
        # grok-3: $3/M in, $15/M out
        cost = calculate_cost("grok-3", 1_000_000, 500_000, reasoning_tokens=500_000)
        # 1M in * $3 + (500k out + 500k reasoning) * $15
        assert cost == 3.00 + 15.00

    def test_calculate_cost_zero_tokens(self):
        """Zero tokens should return zero cost."""
        assert calculate_cost("grok-3", 0, 0) == 0.0

    def test_calculate_image_cost_known_model(self):
        assert calculate_image_cost("grok-imagine-image") == 0.02
        assert calculate_image_cost("grok-imagine-image-pro") == 0.07

    def test_calculate_image_cost_unknown_model(self):
        assert calculate_image_cost("unknown") == 0.07

    def test_calculate_cost_with_cached_tokens(self):
        """Cached tokens should be billed at the discounted rate."""
        # grok-3: $3/M in, $0.75/M cached, $15/M out
        # 1M input with 500k cached: 500k * $3 + 500k * $0.75 + 0 out
        cost = calculate_cost("grok-3", 1_000_000, 0, cached_tokens=500_000)
        assert cost == pytest.approx(1.50 + 0.375)

    def test_calculate_cost_all_cached(self):
        """If all input tokens are cached, only cached rate applies."""
        # grok-3: $0.75/M cached
        cost = calculate_cost("grok-3", 1_000_000, 0, cached_tokens=1_000_000)
        assert cost == pytest.approx(0.75)

    def test_model_pricing_has_three_values(self):
        """Each MODEL_PRICING entry should be a 3-tuple (input, cached, output)."""
        for model, prices in MODEL_PRICING.items():
            assert len(prices) == 3, f"{model} pricing should have 3 values"

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
        cost = calculate_tool_cost({
            "SERVER_SIDE_TOOL_WEB_SEARCH": 1000,
            "SERVER_SIDE_TOOL_COLLECTIONS_SEARCH": 1000,
        })
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
        # 1M chars at $4.20/M = $4.20
        cost = calculate_tts_cost(1_000_000)
        assert cost == pytest.approx(TTS_PRICING_PER_MILLION_CHARS)

    def test_calculate_tts_cost_small(self):
        """Small TTS should be proportional."""
        # 100 chars at $4.20/M
        cost = calculate_tts_cost(100)
        assert cost == pytest.approx(100 / 1_000_000 * TTS_PRICING_PER_MILLION_CHARS)

    def test_calculate_tts_cost_zero(self):
        """Zero characters should return zero cost."""
        assert calculate_tts_cost(0) == 0.0

    def test_calculate_video_cost(self):
        assert calculate_video_cost(5) == 5 * VIDEO_PRICING_PER_SECOND
        assert calculate_video_cost(0) == 0.0


class TestConversation:
    """Tests for the Conversation dataclass."""

    def test_conversation_creation(self):
        """Conversation should store params and response ID state."""
        params = ChatCompletionParameters(model="grok-3")
        conv = Conversation(params=params)

        assert conv.params == params
        assert conv.previous_response_id is None
        assert conv.response_id_history == []
        assert conv.file_ids == []

    def test_conversation_with_response_id(self):
        """Conversation should store response ID and history."""
        params = ChatCompletionParameters(model="grok-3")
        conv = Conversation(
            params=params,
            previous_response_id="resp_123",
            response_id_history=["resp_123"],
            file_ids=["file-1", "file-2"],
        )
        assert conv.previous_response_id == "resp_123"
        assert conv.response_id_history == ["resp_123"]
        assert conv.file_ids == ["file-1", "file-2"]

    def test_default_file_ids_isolated(self):
        """Default file_ids list should not be shared across instances."""
        params = ChatCompletionParameters(model="grok-3")
        conv_one = Conversation(params=params)
        conv_one.file_ids.append("file-1")

        conv_two = Conversation(params=params)
        assert conv_two.file_ids == []
        assert conv_one.file_ids is not conv_two.file_ids

    def test_default_response_id_history_isolated(self):
        """Default response_id_history should not be shared across instances."""
        params = ChatCompletionParameters(model="grok-3")
        conv_one = Conversation(params=params)
        conv_one.response_id_history.append("resp_1")

        conv_two = Conversation(params=params)
        assert conv_two.response_id_history == []
        assert conv_one.response_id_history is not conv_two.response_id_history
