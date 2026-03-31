import pytest

from tests.fixtures import MOCK_RESPONSES_API_RESPONSE
from tests.support import make_cog


class TestExtractToolInfo:
    """Tests for extract_tool_info helper."""

    def test_annotations_deduplicates_and_classifies_citations(self):
        from discord_grok.cogs.grok.responses import extract_tool_info

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
                                {
                                    "type": "url_citation",
                                    "url": "collections://collection_1/files/file_1",
                                },
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
        from discord_grok.cogs.grok.responses import extract_tool_info

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
        from discord_grok.cogs.grok.responses import extract_tool_info

        result = extract_tool_info({"output": []})
        assert result["citations"] == []

    def test_no_annotations(self):
        from discord_grok.cogs.grok.responses import extract_tool_info

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


class TestResponseParsing:
    """Tests for response parsing helpers."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

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
        msg = cog._build_user_message(
            [
                "Describe this",
                {
                    "type": "input_image",
                    "image_url": "https://example.com/img.png",
                    "detail": "high",
                },
            ]
        )
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "input_text", "text": "Describe this"}
        assert msg["content"][1]["type"] == "input_image"
