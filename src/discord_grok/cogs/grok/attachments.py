from __future__ import annotations

import aiohttp
from discord import Attachment

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_SIZE = 20 * 1024 * 1024
MAX_FILE_SIZE = 48 * 1024 * 1024


async def fetch_attachment_bytes(cog, attachment: Attachment) -> bytes | None:
    """Fetch raw bytes for a Discord attachment."""
    session = await cog._get_http_session()
    try:
        async with session.get(attachment.url) as response:
            if response.status == 200:
                return await response.read()
            cog.logger.warning(
                "Failed to fetch attachment %s: HTTP %s",
                attachment.url,
                response.status,
            )
    except aiohttp.ClientError as error:
        cog.logger.warning("Error fetching attachment %s: %s", attachment.url, error)
    return None


def unsupported_image_type_error(attachment: Attachment) -> str | None:
    """Return a user-facing error for unsupported image MIME types."""
    content_type = (attachment.content_type or "").lower()
    if not content_type.startswith("image/") or content_type in SUPPORTED_IMAGE_TYPES:
        return None

    display_type = attachment.content_type or "unknown"
    return (
        f"Unsupported image type `{display_type}` for `{attachment.filename}`. "
        "xAI image input currently supports only JPEG and PNG. Convert the image "
        "to PNG or JPEG and try again."
    )


def build_user_message(content_parts: list[object]) -> dict[str, object]:
    """Build a user message from text and/or multimodal content parts."""
    if len(content_parts) == 1 and isinstance(content_parts[0], str):
        return {"role": "user", "content": content_parts[0]}

    content: list[dict[str, object]] = []
    for part in content_parts:
        if isinstance(part, str):
            content.append({"type": "input_text", "text": part})
        elif isinstance(part, dict):
            content.append(part)
    return {"role": "user", "content": content}


__all__ = [
    "MAX_FILE_SIZE",
    "MAX_IMAGE_SIZE",
    "SUPPORTED_IMAGE_TYPES",
    "build_user_message",
    "fetch_attachment_bytes",
    "unsupported_image_type_error",
]
