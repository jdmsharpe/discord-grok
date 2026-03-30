"""Reusable fixtures for the Grok tests."""

MOCK_RESPONSES_API_RESPONSE: dict = {
    "id": "resp_01XFDUDYJgAACzvnptvVoYEL",
    "output": [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello! How can I help you today?",
                    "annotations": [],
                }
            ],
        }
    ],
    "usage": {
        "input_tokens": 25,
        "output_tokens": 50,
        "total_tokens": 75,
        "input_tokens_details": {
            "cached_tokens": 0,
            "image_tokens": 0,
            "text_tokens": 25,
        },
        "output_tokens_details": {
            "reasoning_tokens": 0,
        },
    },
    "server_side_tool_usage": {},
}

__all__ = ["MOCK_RESPONSES_API_RESPONSE"]
