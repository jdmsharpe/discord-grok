from src.util import (
    CHUNK_TEXT_SIZE,
    ChatCompletionParameters,
    Conversation,
    chunk_text,
    format_xai_error,
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
        assert params.system is None
        assert params.temperature is None
        assert params.top_p is None
        assert params.max_tokens == 16384
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.seed is None
        assert params.reasoning_effort is None
        assert params.paused is False
        assert params.conversation_id is None
        assert params.channel_id is None

    def test_all_params_set(self):
        """All parameters should be stored correctly."""
        params = ChatCompletionParameters(
            model="grok-4-1-fast-reasoning",
            system="You are helpful.",
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=42,
            reasoning_effort="high",
        )
        assert params.model == "grok-4-1-fast-reasoning"
        assert params.system == "You are helpful."
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 2048
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.3
        assert params.seed == 42
        assert params.reasoning_effort == "high"


class TestConversation:
    """Tests for the Conversation dataclass."""

    def test_conversation_creation(self):
        """Conversation should store params and chat object."""
        params = ChatCompletionParameters(model="grok-3")
        mock_chat = object()
        conv = Conversation(params=params, chat=mock_chat)

        assert conv.params == params
        assert conv.chat is mock_chat
