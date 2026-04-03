import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from discord_grok.cogs.grok.client import XaiApiError
from tests.support import MockHTTPSession, make_cog, make_http_response, make_raw_cog


class TestFileUploadAndCleanup:
    """Tests for the xAI Files API integration."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client):
        """Create a cog with files API mocked."""
        cog = make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    async def test_upload_file_attachment_success(self, cog, mock_file_attachment):
        """Should download from Discord and upload to xAI, returning the file ID."""
        with patch.object(cog, "_fetch_attachment_bytes", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = b"file content"

            file_id = await cog._upload_file_attachment(mock_file_attachment)

        assert file_id == "file-abc123"
        cog.client.files.upload.assert_awaited_once_with(b"file content", filename="document.pdf")

    async def test_upload_file_attachment_too_large(self, cog, mock_file_attachment):
        """Files exceeding 48 MB should be rejected."""
        mock_file_attachment.size = 50 * 1024 * 1024

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
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

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
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

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
        from discord_grok.cogs.grok.tooling import ChatCompletionParameters, Conversation

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


class TestSessionManagement:
    """Tests for shared aiohttp session and timeout configuration."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_raw_cog(mock_bot)

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


class TestGrokHTTPRetries:
    """Tests for shared retry behavior across xAI HTTP endpoints."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_raw_cog(mock_bot)

    async def test_call_responses_api_retries_429_with_retry_after(self, cog):
        session = MockHTTPSession(
            [
                make_http_response(
                    429,
                    "rate limited",
                    headers={"Retry-After": "3"},
                ),
                make_http_response(
                    200,
                    json.dumps({"id": "resp_retry", "output": []}),
                ),
            ]
        )

        with (
            patch.object(
                cog,
                "_get_http_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "discord_grok.cogs.grok.client.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            response = await cog._call_responses_api(
                {"model": "grok-3"},
                grok_conv_id="conv-cache-123",
            )

        assert response["id"] == "resp_retry"
        mock_sleep.assert_awaited_once_with(3.0)
        assert len(session.post_calls) == 2
        headers = session.post_calls[0]["kwargs"]["headers"]
        assert headers["x-grok-conv-id"] == "conv-cache-123"

    async def test_call_responses_api_retries_503_with_backoff(self, cog):
        session = MockHTTPSession(
            [
                make_http_response(503, "try again later"),
                make_http_response(
                    200,
                    json.dumps({"id": "resp_ok", "output": []}),
                ),
            ]
        )

        with (
            patch.object(
                cog,
                "_get_http_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "discord_grok.cogs.grok.client.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
            patch("discord_grok.cogs.grok.client.random.uniform", return_value=0.0),
        ):
            response = await cog._call_responses_api({"model": "grok-3"})

        assert response["id"] == "resp_ok"
        mock_sleep.assert_awaited_once_with(0.5)
        assert len(session.post_calls) == 2

    async def test_call_responses_api_fails_fast_for_422(self, cog):
        session = MockHTTPSession([make_http_response(422, "bad input")])

        with (
            patch.object(
                cog,
                "_get_http_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "discord_grok.cogs.grok.client.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
            pytest.raises(XaiApiError, match="bad input"),
        ):
            await cog._call_responses_api({"model": "grok-3"})

        mock_sleep.assert_not_awaited()
        assert len(session.post_calls) == 1
        headers = session.post_calls[0]["kwargs"]["headers"]
        assert "x-grok-conv-id" not in headers

    async def test_call_tts_api_retries_timeouts(self, cog):
        session = MockHTTPSession(
            [
                asyncio.TimeoutError("timed out"),
                make_http_response(200, b"audio-bytes"),
            ]
        )

        with (
            patch.object(
                cog,
                "_get_http_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "discord_grok.cogs.grok.client.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
            patch("discord_grok.cogs.grok.client.random.uniform", return_value=0.0),
        ):
            audio = await cog._call_tts_api({"text": "Hello"})

        assert audio == b"audio-bytes"
        mock_sleep.assert_awaited_once_with(0.5)
        assert len(session.post_calls) == 2
        headers = session.post_calls[0]["kwargs"]["headers"]
        assert "x-grok-conv-id" not in headers
