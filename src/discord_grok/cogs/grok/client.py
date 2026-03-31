import asyncio
import contextlib
import json
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, cast

import aiohttp
import xai_sdk
from aiohttp import ClientTimeout

from ...config.auth import XAI_API_KEY
from .attachments import MAX_FILE_SIZE

TTS_API_URL = "https://api.x.ai/v1/tts"
TTS_MAX_CHARS = 15_000
RESPONSES_API_URL = "https://api.x.ai/v1/responses"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_API_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 0.5
RETRY_JITTER_RATIO = 0.25


def get_client(cog) -> Any:
    if cog.client is None:
        cog.client = xai_sdk.AsyncClient(api_key=XAI_API_KEY or None)
    return cog.client


async def get_http_session(cog) -> aiohttp.ClientSession:
    if cog._http_session and not cog._http_session.closed:
        return cog._http_session
    async with cog._session_lock:
        if cog._http_session is None or cog._http_session.closed:
            cog._http_session = aiohttp.ClientSession(timeout=ClientTimeout(total=300, connect=15))
        return cog._http_session


def build_xai_headers(*, grok_conv_id: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if grok_conv_id:
        headers["x-grok-conv-id"] = grok_conv_id
    return headers


def describe_api_request(url: str) -> str:
    if url == RESPONSES_API_URL:
        return "Responses API"
    if url == TTS_API_URL:
        return "TTS API"
    return "xAI API"


def parse_retry_after(retry_after: str | None) -> float | None:
    if not retry_after:
        return None
    retry_after = retry_after.strip()
    with contextlib.suppress(ValueError):
        return max(0.0, float(retry_after))
    with contextlib.suppress(TypeError, ValueError, OverflowError):
        retry_at = parsedate_to_datetime(retry_after)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    return None


def compute_retry_delay(attempt: int, *, retry_after: str | None = None) -> float:
    parsed_retry_after = parse_retry_after(retry_after)
    if parsed_retry_after is not None:
        return parsed_retry_after
    base_delay = INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
    return base_delay + random.uniform(0.0, base_delay * RETRY_JITTER_RATIO)


async def post_with_retries(
    cog,
    url: str,
    headers: dict[str, str],
    json_payload: dict[str, Any],
) -> bytes:
    session = await cog._get_http_session()
    request_name = describe_api_request(url)

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            async with session.post(url, headers=headers, json=json_payload) as resp:
                if resp.status == 200:
                    return await resp.read()
                error_body = await resp.text()
                should_retry = resp.status in RETRYABLE_STATUS_CODES and attempt < MAX_API_ATTEMPTS
                if should_retry:
                    delay = compute_retry_delay(
                        attempt,
                        retry_after=(
                            resp.headers.get("Retry-After") if resp.status == 429 else None
                        ),
                    )
                    cog.logger.warning(
                        "%s request returned HTTP %s on attempt %d/%d; retrying in %.2fs",
                        request_name,
                        resp.status,
                        attempt,
                        MAX_API_ATTEMPTS,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise Exception(f"{request_name} error (HTTP {resp.status}): {error_body}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            if attempt >= MAX_API_ATTEMPTS:
                raise Exception(
                    f"{request_name} request failed after {MAX_API_ATTEMPTS} attempts: {error}"
                ) from error
            delay = compute_retry_delay(attempt)
            cog.logger.warning(
                "%s request failed on attempt %d/%d (%s); retrying in %.2fs",
                request_name,
                attempt,
                MAX_API_ATTEMPTS,
                error,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"{request_name} retry loop exited unexpectedly.")


async def generate_tts(
    cog,
    text: str,
    voice_id: str,
    language: str,
    codec: str,
    sample_rate: int | None = None,
    bit_rate: int | None = None,
) -> bytes:
    output_format: dict[str, Any] = {"codec": codec}
    if sample_rate is not None:
        output_format["sample_rate"] = sample_rate
    if bit_rate is not None and codec == "mp3":
        output_format["bit_rate"] = bit_rate
    payload: dict[str, Any] = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
        "output_format": output_format,
    }
    return await call_tts_api(cog, payload)


async def call_tts_api(cog, payload: dict[str, Any]) -> bytes:
    return await post_with_retries(
        cog,
        TTS_API_URL,
        headers=build_xai_headers(),
        json_payload=payload,
    )


async def call_responses_api(
    cog,
    payload: dict[str, Any],
    *,
    grok_conv_id: str | None = None,
) -> dict[str, Any]:
    response_body = await post_with_retries(
        cog,
        RESPONSES_API_URL,
        headers=build_xai_headers(grok_conv_id=grok_conv_id),
        json_payload=payload,
    )
    return cast(dict[str, Any], json.loads(response_body))


def build_responses_payload(
    model: str,
    input_messages: list[dict[str, Any]],
    *,
    previous_response_id: str | None = None,
    prompt_cache_key: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: str | None = None,
    agent_count: int | None = None,
    include_encrypted_reasoning: bool = False,
) -> dict[str, Any]:
    """Build a JSON payload for the xAI Responses API."""
    payload: dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "store": True,
    }
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if tools:
        payload["tools"] = tools
    if include_encrypted_reasoning:
        payload["include"] = ["reasoning.encrypted_content"]
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if agent_count is not None:
        payload["agent_count"] = agent_count
    return payload


async def upload_file_attachment(cog, attachment, *, fetch_bytes) -> str | None:
    if attachment.size > MAX_FILE_SIZE:
        cog.logger.warning(
            "Attachment %s exceeds 48 MB limit (%s bytes)",
            attachment.filename,
            attachment.size,
        )
        return None
    file_bytes = await fetch_bytes(attachment)
    if file_bytes is None:
        return None
    client = get_client(cog)
    try:
        uploaded = await client.files.upload(file_bytes, filename=attachment.filename)
        cog.logger.info("Uploaded file %s as %s", attachment.filename, uploaded.id)
        return uploaded.id
    except Exception as error:
        cog.logger.warning("Failed to upload file %s to xAI: %s", attachment.filename, error)
        return None


async def cleanup_conversation_files(cog, conversation) -> None:
    if not conversation.file_ids:
        return
    client = get_client(cog)
    for file_id in conversation.file_ids:
        try:
            await client.files.delete(file_id)
            cog.logger.info("Deleted xAI file %s", file_id)
        except Exception as error:
            cog.logger.warning("Failed to delete xAI file %s: %s", file_id, error)
    conversation.file_ids.clear()


async def close_http_session(cog) -> None:
    session = cog._http_session
    if session and not session.closed:
        try:
            await session.close()
        except Exception as error:
            cog.logger.warning("Error closing HTTP session: %s", error)
    cog._http_session = None


__all__ = [
    "ClientTimeout",
    "INITIAL_RETRY_DELAY_SECONDS",
    "MAX_API_ATTEMPTS",
    "RESPONSES_API_URL",
    "RETRYABLE_STATUS_CODES",
    "RETRY_JITTER_RATIO",
    "TTS_API_URL",
    "TTS_MAX_CHARS",
    "build_xai_headers",
    "build_responses_payload",
    "call_responses_api",
    "call_tts_api",
    "cleanup_conversation_files",
    "close_http_session",
    "compute_retry_delay",
    "describe_api_request",
    "generate_tts",
    "get_client",
    "get_http_session",
    "parse_retry_after",
    "post_with_retries",
    "upload_file_attachment",
]
