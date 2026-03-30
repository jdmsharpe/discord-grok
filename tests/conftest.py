from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot instance."""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 123456789
    bot.owner_id = 987654321
    return bot


@pytest.fixture
def mock_xai_client():
    """Create a mock xAI SDK AsyncClient (files, image, video only)."""
    with patch("xai_sdk.AsyncClient") as mock_class:
        client = MagicMock()

        # Mock image.sample
        mock_image_response = MagicMock()
        mock_image_response.url = "https://example.com/generated-image.png"
        mock_image_response.base64 = None
        client.image = MagicMock()
        client.image.sample = AsyncMock(return_value=mock_image_response)
        client.image.sample_batch = AsyncMock(
            return_value=[mock_image_response, mock_image_response]
        )

        # Mock video.generate
        mock_video_response = MagicMock()
        mock_video_response.url = "https://example.com/generated-video.mp4"
        client.video = MagicMock()
        client.video.generate = AsyncMock(return_value=mock_video_response)

        # Mock files API
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "file-abc123"
        mock_uploaded_file.filename = "document.pdf"
        mock_uploaded_file.size = 1024
        client.files = MagicMock()
        client.files.upload = AsyncMock(return_value=mock_uploaded_file)
        mock_delete_response = MagicMock()
        mock_delete_response.deleted = True
        mock_delete_response.id = "file-abc123"
        client.files.delete = AsyncMock(return_value=mock_delete_response)

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
    """Create a mock Discord image attachment."""
    attachment = MagicMock()
    attachment.url = "https://example.com/image.png"
    attachment.content_type = "image/png"
    attachment.filename = "image.png"
    attachment.size = 1024
    return attachment


@pytest.fixture
def mock_file_attachment():
    """Create a mock Discord file attachment (non-image)."""
    attachment = MagicMock()
    attachment.url = "https://example.com/document.pdf"
    attachment.content_type = "application/pdf"
    attachment.filename = "document.pdf"
    attachment.size = 2048
    return attachment
