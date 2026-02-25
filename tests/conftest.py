import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src/ to path so imports like "from button_view import ..." work in Docker
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot instance."""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 123456789
    bot.owner_id = 987654321
    bot.sync_commands = AsyncMock()
    return bot


@pytest.fixture
def mock_xai_client():
    """Create a mock xAI SDK AsyncClient."""
    with patch("xai_sdk.AsyncClient") as mock_class:
        client = MagicMock()

        # Mock chat.create -> Chat object -> sample() -> Response
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello! How can I help you today?"
        mock_response.reasoning_content = ""
        mock_response.citations = []
        mock_response.server_side_tool_usage = {}
        mock_response.tool_calls = []
        mock_response.id = "resp_01XFDUDYJgAACzvnptvVoYEL"
        mock_response.role = "assistant"
        mock_response.finish_reason = "stop"

        mock_chat.sample = AsyncMock(return_value=mock_response)
        mock_chat.append = MagicMock()
        mock_chat.messages = []

        client.chat = MagicMock()
        client.chat.create = MagicMock(return_value=mock_chat)

        # Mock image.sample
        mock_image_response = MagicMock()
        mock_image_response.url = "https://example.com/generated-image.png"
        mock_image_response.base64 = None
        client.image = MagicMock()
        client.image.sample = AsyncMock(return_value=mock_image_response)

        # Mock video.generate
        mock_video_response = MagicMock()
        mock_video_response.url = "https://example.com/generated-video.mp4"
        client.video = MagicMock()
        client.video.generate = AsyncMock(return_value=mock_video_response)

        mock_class.return_value = client
        yield client


@pytest.fixture
def mock_discord_context():
    """Create a mock Discord application context."""
    ctx = AsyncMock()
    ctx.author = MagicMock()
    ctx.author.id = 111222333
    ctx.author.name = "TestUser"
    ctx.channel = MagicMock()
    ctx.channel.id = 444555666
    ctx.interaction = MagicMock()
    ctx.interaction.id = 777888999
    ctx.defer = AsyncMock()
    ctx.send_followup = AsyncMock()
    ctx.respond = AsyncMock()
    return ctx


@pytest.fixture
def mock_discord_message():
    """Create a mock Discord message."""
    message = MagicMock()
    message.author = MagicMock()
    message.author.id = 111222333
    message.author.name = "TestUser"
    message.channel = MagicMock()
    message.channel.id = 444555666
    message.content = "Hello Grok!"
    message.attachments = []
    message.reply = AsyncMock()
    return message


@pytest.fixture
def mock_attachment():
    """Create a mock Discord attachment."""
    attachment = MagicMock()
    attachment.url = "https://example.com/image.png"
    attachment.content_type = "image/png"
    attachment.filename = "image.png"
    return attachment
