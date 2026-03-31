from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.grok_test_support import make_cog


class TestGrokCommandSchema:
    """Tests for command option wiring and choices."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

    def test_chat_model_choices_match_grok_models(self, cog):
        """Chat command model choices should match GROK_MODELS."""
        from discord_grok.cogs.grok.tooling import GROK_MODELS

        chat_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "chat")
        model_option = next(opt for opt in chat_cmd.options if opt.name == "model")
        choice_values = sorted(choice.value for choice in model_option.choices)
        assert choice_values == sorted(GROK_MODELS)

    def test_image_model_choices_match_grok_image_models(self, cog):
        """Image command model choices should match GROK_IMAGE_MODELS."""
        from discord_grok.cogs.grok.tooling import GROK_IMAGE_MODELS

        image_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "image")
        model_option = next(opt for opt in image_cmd.options if opt.name == "model")
        choice_values = sorted(choice.value for choice in model_option.choices)
        assert choice_values == sorted(GROK_IMAGE_MODELS)

    def test_image_aspect_ratios_match_sdk(self, cog):
        """Image command aspect ratios should match xai-sdk ImageAspectRatio."""
        from xai_sdk.image import ImageAspectRatio

        image_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "image")
        ar_option = next(opt for opt in image_cmd.options if opt.name == "aspect_ratio")
        choice_values = sorted(choice.value for choice in ar_option.choices)
        assert choice_values == sorted(ImageAspectRatio.__args__)

    def test_image_resolution_choices_match_sdk(self, cog):
        """Image command resolution choices should match xai-sdk ImageResolution."""
        from xai_sdk.image import ImageResolution

        image_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "image")
        res_option = next(opt for opt in image_cmd.options if opt.name == "resolution")
        choice_values = sorted(choice.value for choice in res_option.choices)
        assert choice_values == sorted(ImageResolution.__args__)

    def test_image_has_attachment_option(self, cog):
        """Image command should have an optional attachment parameter."""
        image_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "image")
        att_option = next((opt for opt in image_cmd.options if opt.name == "attachment"), None)
        assert att_option is not None
        assert att_option.required is False

    def test_video_aspect_ratios_match_sdk(self, cog):
        """Video command aspect ratios should match xai-sdk VideoAspectRatio."""
        from xai_sdk.video import VideoAspectRatio

        video_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "video")
        ar_option = next(opt for opt in video_cmd.options if opt.name == "aspect_ratio")
        choice_values = sorted(choice.value for choice in ar_option.choices)
        assert choice_values == sorted(VideoAspectRatio.__args__)

    def test_video_has_attachment_option(self, cog):
        """Video command should have an optional attachment parameter."""
        video_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "video")
        att_option = next((opt for opt in video_cmd.options if opt.name == "attachment"), None)
        assert att_option is not None
        assert att_option.required is False

    def test_tts_voice_choices_match_tts_voices(self, cog):
        """TTS command voice choices should match TTS_VOICES."""
        from discord_grok.cogs.grok.tooling import TTS_VOICES

        tts_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "tts")
        voice_option = next(opt for opt in tts_cmd.options if opt.name == "voice")
        choice_values = sorted(choice.value for choice in voice_option.choices)
        assert choice_values == sorted(TTS_VOICES)


class TestTTSCommand:
    """Tests for the /grok tts command."""

    @pytest.fixture
    def cog(self, mock_bot):
        with patch("xai_sdk.AsyncClient"):
            from discord_grok import GrokCog

            return GrokCog(bot=mock_bot)

    async def test_tts_text_too_long(self, cog, mock_discord_context):
        """Text over 15,000 chars should be rejected."""
        await cog.tts.callback(
            cog,
            ctx=mock_discord_context,
            text="a" * 15001,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "15,000" in call_kwargs["embed"].description

    async def test_tts_success(self, cog, mock_discord_context):
        """Successful TTS should send an audio file with metadata embed."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"fake audio bytes"

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hello world",
                voice="eve",
                language="en",
                output_format="mp3",
            )

        mock_gen.assert_awaited_once_with("Hello world", "eve", "en", "mp3", None, None)
        mock_discord_context.send_followup.assert_called_once()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embeds"][0].title == "Text-to-Speech Generation"
        assert call_kwargs["file"] is not None

    async def test_tts_with_sample_rate_and_bit_rate(self, cog, mock_discord_context):
        """sample_rate and bit_rate should be forwarded to _generate_tts."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"fake audio bytes"

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hi",
                voice="rex",
                language="auto",
                output_format="mp3",
                sample_rate=44100,
                bit_rate=192000,
            )

        mock_gen.assert_awaited_once_with("Hi", "rex", "auto", "mp3", 44100, 192000)
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "44,100 Hz" in call_kwargs["embeds"][0].description
        assert "192 kbps" in call_kwargs["embeds"][0].description

    async def test_tts_mulaw_file_extension(self, cog, mock_discord_context):
        """mulaw codec should produce a .ulaw file extension."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = b"fake audio bytes"

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hello",
                output_format="mulaw",
            )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["file"].filename == "speech.ulaw"

    async def test_tts_api_error(self, cog, mock_discord_context):
        """API errors should display an error embed."""
        with patch.object(cog, "_generate_tts", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("TTS API error (HTTP 400): bad request")

            await cog.tts.callback(
                cog,
                ctx=mock_discord_context,
                text="Hello",
            )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embed"].title == "Error"


class TestImageBatchGeneration:
    """Tests for multi-image generation via sample_batch."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client):
        """Create a cog with xAI image SDK mocked."""
        cog = make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    @staticmethod
    def _mock_http_session():
        """Create a mock HTTP session with a working async context manager for get()."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"fake image bytes")

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm
        return mock_session

    def test_image_has_count_option(self, cog):
        """Image command should have a count parameter with min=1, max=10."""
        image_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "image")
        count_option = next((opt for opt in image_cmd.options if opt.name == "count"), None)
        assert count_option is not None
        assert count_option.required is False
        assert count_option.min_value == 1
        assert count_option.max_value == 10

    async def test_image_single_calls_sample(self, cog, mock_discord_context):
        """count=1 should call client.image.sample (not sample_batch)."""
        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image-pro",
                count=1,
            )

        cog.client.image.sample.assert_awaited_once()
        cog.client.image.sample_batch.assert_not_awaited()

    async def test_image_batch_calls_sample_batch(self, cog, mock_discord_context):
        """count>1 should call client.image.sample_batch with n=count."""
        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image-pro",
                count=3,
            )

        cog.client.image.sample.assert_not_awaited()
        cog.client.image.sample_batch.assert_awaited_once()
        call_kwargs = cog.client.image.sample_batch.call_args
        assert call_kwargs.kwargs["n"] == 3

    async def test_image_batch_sends_multiple_files(self, cog, mock_discord_context):
        """Batch generation should send multiple File objects."""
        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image",
                count=2,
            )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        files = call_kwargs["files"]
        assert len(files) == 2
        assert files[0].filename == "image_1.png"
        assert files[1].filename == "image_2.png"

    async def test_image_batch_cost_multiplied(self, cog, mock_discord_context):
        """Batch generation cost should be per-image cost × count."""
        from discord_grok.cogs.grok.tooling import calculate_image_cost

        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.image.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A cat",
                model="grok-imagine-image",
                count=2,
            )

        expected_cost = calculate_image_cost("grok-imagine-image") * 2
        from datetime import date

        key = (mock_discord_context.author.id, date.today().isoformat())
        assert abs(cog.daily_costs[key] - expected_cost) < 1e-9

    async def test_image_batch_rejects_editing_mode(
        self, cog, mock_discord_context, mock_attachment
    ):
        """count>1 with an attachment (editing mode) should return an error."""
        await cog.image.callback(
            cog,
            ctx=mock_discord_context,
            prompt="Edit this",
            model="grok-imagine-image-pro",
            count=3,
            attachment=mock_attachment,
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert "not supported in Image Editing mode" in call_kwargs["embed"].description
        cog.client.image.sample.assert_not_awaited()
        cog.client.image.sample_batch.assert_not_awaited()


class TestVideoCommand:
    """Integration tests for the /grok video command."""

    @pytest.fixture
    def cog(self, mock_bot, mock_xai_client):
        cog = make_cog(mock_bot)
        cog.client = mock_xai_client
        return cog

    @staticmethod
    def _mock_http_session():
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=b"fake video bytes")

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_cm
        return mock_session

    async def test_video_success(self, cog, mock_discord_context):
        """Successful text-to-video should send a video file."""
        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.video.callback(
                cog,
                ctx=mock_discord_context,
                prompt="A sunset",
            )

        cog.client.video.generate.assert_awaited_once()
        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["file"].filename == "video.mp4"

    async def test_video_with_attachment(self, cog, mock_discord_context, mock_attachment):
        """Image-to-video should pass image_url to the SDK."""
        with patch.object(
            cog,
            "_get_http_session",
            new_callable=AsyncMock,
            return_value=self._mock_http_session(),
        ):
            await cog.video.callback(
                cog,
                ctx=mock_discord_context,
                prompt="Animate this",
                attachment=mock_attachment,
            )

        gen_kwargs = cog.client.video.generate.call_args[1]
        assert gen_kwargs["image_url"] == str(mock_attachment.url)

    async def test_video_api_error(self, cog, mock_discord_context):
        """API errors should display an error embed."""
        cog.client.video.generate.side_effect = Exception("Video gen failed")

        await cog.video.callback(
            cog,
            ctx=mock_discord_context,
            prompt="A sunset",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embed"].title == "Error"

    async def test_video_no_url_returns_error(self, cog, mock_discord_context):
        """No video URL from API should display an error."""
        cog.client.video.generate.return_value.url = None

        await cog.video.callback(
            cog,
            ctx=mock_discord_context,
            prompt="A sunset",
        )

        call_kwargs = mock_discord_context.send_followup.call_args[1]
        assert call_kwargs["embed"].title == "Error"
