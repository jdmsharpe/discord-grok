import hashlib
import hmac
import inspect
import json
import logging
import sys
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discord_grok.cogs.grok import client as client_module
from discord_grok.cogs.grok import safety
from discord_grok.cogs.grok.client import build_responses_payload, generate_tts
from discord_grok.cogs.grok.safety import (
    SAFETY_IDENTIFIER_KEY_LABEL,
    build_safety_identifier,
    derive_safety_identifier_key,
)
from discord_grok.config import auth
from tests.support import make_cog

USER_ID = 111222333
OTHER_USER_ID = 444555666
TEST_SECRET = "test-safety-secret-value"


@pytest.fixture
def secret_key(monkeypatch):
    key = derive_safety_identifier_key(TEST_SECRET, "unused-bot-token")
    monkeypatch.setattr(safety, "SAFETY_IDENTIFIER_KEY", key)
    return key


def _expected(user_id: int, key: bytes) -> str:
    return hmac.new(key, str(user_id).encode(), hashlib.sha256).hexdigest()


def _load_safety_with_env(monkeypatch, env: dict[str, str | None]):
    """Run auth.py and safety.py as fresh modules under ``env``.

    The fresh auth module replaces ``discord_grok.config.auth`` in
    ``sys.modules`` only for the test, so the relative import in safety.py
    reads it; ``load_dotenv`` is stubbed so a local .env cannot fill in a
    variable the test leaves unset.
    """
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)

    fresh_auth = _exec_fresh_module("discord_grok.config.auth", auth.__file__)
    monkeypatch.setitem(sys.modules, "discord_grok.config.auth", fresh_auth)
    return _exec_fresh_module("discord_grok.cogs.grok.safety", safety.__file__)


def _exec_fresh_module(name: str, path: str | None) -> ModuleType:
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_http_session(body: bytes):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read = AsyncMock(return_value=body)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get.return_value = mock_cm
    return mock_session


def _typing_ctx(ctx):
    ctx.channel.typing = MagicMock()
    ctx.channel.typing.return_value.__aenter__ = AsyncMock()
    ctx.channel.typing.return_value.__aexit__ = AsyncMock()
    return ctx


class TestKeyDerivation:
    def test_secret_env_var_is_the_key_when_set(self):
        assert derive_safety_identifier_key("abc", "bot-token") == b"abc"

    def test_bot_token_fallback_uses_hmac_with_the_label(self):
        key = derive_safety_identifier_key(None, "bot-token")
        assert key == hmac.new(b"bot-token", SAFETY_IDENTIFIER_KEY_LABEL, hashlib.sha256).digest()
        assert key != b"bot-token"

    def test_bot_token_fallback_uses_the_shared_label(self):
        """discord-claude, discord-openai and discord-openrouter derive the key the same
        way, so a user gets the same value from every bot that shares a BOT_TOKEN or
        secret."""
        assert SAFETY_IDENTIFIER_KEY_LABEL == b"safety-identifier-v1"
        key = derive_safety_identifier_key(None, "bot-token")
        assert key is not None
        expected_key = hmac.new(b"bot-token", b"safety-identifier-v1", hashlib.sha256).digest()
        assert _expected(USER_ID, key) == _expected(USER_ID, expected_key)

    @pytest.mark.parametrize(
        ("secret", "bot_token", "expected"),
        [
            pytest.param(
                None,
                "tok",
                "e4c4e9a56e63186621396e88847c96bf5ebd4caae31c5b1097c7933eb939c85e",
                id="bot-token-fallback",
            ),
            pytest.param(
                "shared-secret",
                "tok",
                "1f4ae57d70d012644d5b186e086ba1651888e8dbb7cc311b6f9fefee5dc8862f",
                id="secret-set",
            ),
        ],
    )
    def test_fixed_vectors_shared_with_sibling_bots(self, monkeypatch, secret, bot_token, expected):
        """Literal values for the construction shared with discord-claude, discord-openai
        and discord-openrouter, so a change to how the key or the user ID is encoded
        fails here instead of only splitting one user into different values."""
        key = derive_safety_identifier_key(secret, bot_token)
        monkeypatch.setattr(safety, "SAFETY_IDENTIFIER_KEY", key)
        assert build_safety_identifier(111222333) == expected
        assert build_safety_identifier("111222333") == expected

    def test_no_key_without_secret_or_bot_token(self):
        assert derive_safety_identifier_key(None, None) is None

    @pytest.mark.parametrize(
        ("secret", "bot_token", "expected_key"),
        [
            pytest.param("env-secret", "env-bot-token", b"env-secret", id="secret-set"),
            pytest.param(
                None,
                "env-bot-token",
                hmac.new(b"env-bot-token", SAFETY_IDENTIFIER_KEY_LABEL, hashlib.sha256).digest(),
                id="bot-token-fallback",
            ),
            pytest.param(None, None, None, id="neither-set"),
        ],
    )
    def test_module_key_is_read_from_the_environment(
        self, monkeypatch, secret, bot_token, expected_key
    ):
        fresh_safety = _load_safety_with_env(
            monkeypatch,
            {
                "SAFETY_IDENTIFIER_SECRET": secret,
                "BOT_TOKEN": bot_token,
                "XAI_API_KEY": "env-api-key",
            },
        )

        module_key = fresh_safety.SAFETY_IDENTIFIER_KEY
        assert module_key == expected_key
        identifier = fresh_safety.build_safety_identifier(USER_ID)
        if expected_key is None:
            assert identifier is None
        else:
            assert identifier == _expected(USER_ID, expected_key)

    def test_xai_api_key_is_never_an_input(self):
        """xAI holds XAI_API_KEY, so a key derived from it would let xAI rebuild
        the mapping by hashing known Discord user IDs."""
        assert "XAI_API_KEY" not in vars(safety)


class TestBuildSafetyIdentifier:
    def test_is_a_64_char_hex_hmac_of_the_user_id(self, secret_key):
        value = build_safety_identifier(USER_ID)
        assert value == _expected(USER_ID, secret_key)
        assert len(value) == 64
        int(value, 16)

    def test_is_stable_for_a_user_and_secret(self, secret_key):
        assert build_safety_identifier(USER_ID) == build_safety_identifier(USER_ID)
        assert build_safety_identifier(USER_ID) == build_safety_identifier(str(USER_ID))

    def test_differs_across_users(self, secret_key):
        assert build_safety_identifier(USER_ID) != build_safety_identifier(OTHER_USER_ID)

    def test_differs_across_secrets(self, monkeypatch):
        values = set()
        for secret in ("secret-a", "secret-b"):
            monkeypatch.setattr(
                safety, "SAFETY_IDENTIFIER_KEY", derive_safety_identifier_key(secret, None)
            )
            values.add(build_safety_identifier(USER_ID))
        for bot_token in ("token-a", "token-b"):
            monkeypatch.setattr(
                safety, "SAFETY_IDENTIFIER_KEY", derive_safety_identifier_key(None, bot_token)
            )
            values.add(build_safety_identifier(USER_ID))
        assert len(values) == 4

    def test_does_not_reveal_the_user_id(self, secret_key):
        value = build_safety_identifier(USER_ID)
        assert value is not None
        assert str(USER_ID) not in value
        assert format(USER_ID, "x") not in value
        assert value != hashlib.sha256(str(USER_ID).encode()).hexdigest()
        assert value != hashlib.sha256(USER_ID.to_bytes(8, "big")).hexdigest()

    def test_returns_none_without_a_key(self, monkeypatch):
        monkeypatch.setattr(safety, "SAFETY_IDENTIFIER_KEY", None)
        assert build_safety_identifier(USER_ID) is None


class TestResponsesPayload:
    def test_payload_carries_safety_identifier(self):
        payload = build_responses_payload("grok-4.7", [], safety_identifier="ab" * 32)
        assert payload["safety_identifier"] == "ab" * 32

    def test_payload_omits_safety_identifier_when_none(self):
        payload = build_responses_payload("grok-4.7", [])
        assert "safety_identifier" not in payload


class TestChatSendsSafetyIdentifier:
    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

    async def test_chat_request_carries_the_authors_identifier(
        self, cog, mock_discord_context, secret_key
    ):
        await cog.chat.callback(cog, ctx=_typing_ctx(mock_discord_context), prompt="Hello")

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["safety_identifier"] == _expected(mock_discord_context.author.id, secret_key)
        _assert_no_raw_user_id([json.dumps(payload)], mock_discord_context.author.id)

    async def test_follow_up_request_carries_the_authors_identifier(
        self, cog, mock_discord_context, secret_key
    ):
        await cog.chat.callback(cog, ctx=_typing_ctx(mock_discord_context), prompt="Hello")
        conversation = next(iter(cog.conversations.values()))
        cog._call_responses_api.reset_mock()

        message = MagicMock()
        message.author = conversation.params.conversation_starter
        message.channel = _typing_ctx(MagicMock()).channel
        message.content = "Follow-up"
        message.attachments = []
        message.reply = AsyncMock()

        await cog.handle_new_message_in_conversation(message, conversation)

        payload = cog._call_responses_api.call_args[0][0]
        assert payload["safety_identifier"] == _expected(mock_discord_context.author.id, secret_key)
        _assert_no_raw_user_id([json.dumps(payload)], mock_discord_context.author.id)

    async def test_chat_omits_the_field_without_a_key(self, cog, mock_discord_context, monkeypatch):
        monkeypatch.setattr(safety, "SAFETY_IDENTIFIER_KEY", None)

        await cog.chat.callback(cog, ctx=_typing_ctx(mock_discord_context), prompt="Hello")

        payload = cog._call_responses_api.call_args[0][0]
        assert "safety_identifier" not in payload

    async def test_secret_and_identifier_never_reach_the_logs(
        self, cog, mock_discord_context, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            safety, "SAFETY_IDENTIFIER_KEY", derive_safety_identifier_key(TEST_SECRET, None)
        )
        identifier = build_safety_identifier(mock_discord_context.author.id)
        assert identifier is not None

        with caplog.at_level(logging.DEBUG):
            await cog.chat.callback(cog, ctx=_typing_ctx(mock_discord_context), prompt="Hello")

        assert caplog.records
        assert TEST_SECRET not in caplog.text
        assert identifier not in caplog.text


def _assert_no_raw_user_id(values, user_id: int) -> None:
    for value in values:
        assert str(user_id) not in str(value)
        assert format(user_id, "x") not in str(value)


class TestImageRequestsCarryUserIdentifier:
    """The Images API has no `safety_identifier`; xai-sdk 1.20.0 `image.sample`
    and `sample_batch` take the legacy `user` field, which gets the same value."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client):
        cog = make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    @pytest.mark.parametrize(
        ("count", "sdk_method", "edit"),
        [(1, "sample", False), (2, "sample_batch", False), (1, "sample", True)],
        ids=["generate-single", "generate-batch", "edit"],
    )
    async def test_image_sdk_call_carries_the_authors_identifier(
        self, cog, mock_discord_context, mock_attachment, secret_key, count, sdk_method, edit
    ):
        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=_mock_http_session(b"fake image bytes"),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                count=count,
                attachment=mock_attachment if edit else None,
            )

        kwargs = getattr(cog.client.image, sdk_method).await_args.kwargs
        assert kwargs["user"] == _expected(mock_discord_context.author.id, secret_key)
        # The fixture's image client is an unspecced mock, so check the kwargs
        # against the installed xai-sdk method; a renamed or dropped parameter
        # would otherwise raise TypeError only at runtime.
        from xai_sdk.aio.image import Client as AsyncImageClient

        sdk_params = inspect.signature(getattr(AsyncImageClient, sdk_method)).parameters
        assert set(kwargs) <= set(sdk_params)
        assert "safety_identifier" not in kwargs
        assert ("image_url" in kwargs) is edit
        _assert_no_raw_user_id(kwargs.values(), mock_discord_context.author.id)

    async def test_image_sdk_call_omits_user_without_a_key(
        self, cog, mock_discord_context, monkeypatch
    ):
        monkeypatch.setattr(safety, "SAFETY_IDENTIFIER_KEY", None)

        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=_mock_http_session(b"fake image bytes"),
        ):
            await cog.image.callback(cog, ctx=mock_discord_context, prompt="A cat", count=1)

        kwargs = cog.client.image.sample.await_args.kwargs
        assert "user" not in kwargs
        _assert_no_raw_user_id(kwargs.values(), mock_discord_context.author.id)


class TestRequestsWithoutSafetyIdentifier:
    """xai-sdk 1.20.0 `video.generate` has neither `safety_identifier` nor `user`,
    and POST /v1/tts documents neither field."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client, secret_key):
        cog = make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    async def test_video_sdk_call_has_no_user_identifier(self, cog, mock_discord_context):
        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=_mock_http_session(b"fake video bytes"),
        ):
            await cog.video.callback(cog, ctx=mock_discord_context, prompt="A sunset")

        kwargs = cog.client.video.generate.await_args.kwargs
        assert "safety_identifier" not in kwargs
        assert "user" not in kwargs
        _assert_no_raw_user_id(kwargs.values(), mock_discord_context.author.id)

    async def test_tts_payload_has_no_safety_identifier(self, cog):
        with patch.object(
            client_module, "call_tts_api", new_callable=AsyncMock, return_value=b"audio"
        ) as mock_call:
            await generate_tts(cog, "Hello", "eve", "en", "mp3")

        payload = mock_call.await_args.args[1]
        assert "safety_identifier" not in payload
