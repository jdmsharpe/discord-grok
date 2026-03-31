import copy
from unittest.mock import AsyncMock, MagicMock, patch

from tests.fixtures import MOCK_RESPONSES_API_RESPONSE


def make_raw_cog(mock_bot):
    """Helper to create a GrokCog without overriding runtime methods."""
    with patch("xai_sdk.AsyncClient"):
        from discord_grok import GrokCog

        return GrokCog(bot=mock_bot)


def make_cog(mock_bot, mock_api_response=None):
    """Helper to create a cog with _call_responses_api mocked."""
    cog = make_raw_cog(mock_bot)

    if mock_api_response is None:
        mock_api_response = copy.deepcopy(MOCK_RESPONSES_API_RESPONSE)
    cog._call_responses_api = AsyncMock(return_value=mock_api_response)
    return cog


def make_http_response(
    status: int,
    body: str | bytes,
    headers: dict[str, str] | None = None,
):
    response = MagicMock()
    response.status = status
    response.headers = headers or {}
    response.read = AsyncMock(return_value=body if isinstance(body, bytes) else body.encode())
    response.text = AsyncMock(return_value=body.decode() if isinstance(body, bytes) else body)
    return response


class MockPostContextManager:
    def __init__(self, *, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc

    async def __aenter__(self):
        if self.exc is not None:
            raise self.exc
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockHTTPSession:
    def __init__(self, post_results: list[object]):
        self.post_results = list(post_results)
        self.post_calls: list[dict[str, object]] = []

    def post(self, *args, **kwargs):
        self.post_calls.append({"args": args, "kwargs": kwargs})
        result = self.post_results.pop(0)
        if isinstance(result, Exception):
            return MockPostContextManager(exc=result)
        return MockPostContextManager(response=result)
