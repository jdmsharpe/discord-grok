from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.support import make_cog


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

    def test_chat_model_choices_exist_in_shared_metadata_with_pricing(self, cog):
        """Every chat slash-choice model should exist in shared metadata with pricing."""
        from discord_grok.cogs.grok.command_options import CHAT_MODEL_INDEX
        from discord_grok.cogs.grok.tooling import MODEL_PRICING

        chat_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "chat")
        model_option = next(opt for opt in chat_cmd.options if opt.name == "model")

        for choice in model_option.choices:
            assert choice.value in CHAT_MODEL_INDEX
            assert choice.value in MODEL_PRICING

    def test_default_chat_model_is_grok_4_6(self):
        """grok-4.6 was promoted to default. grok-4.5 and grok-4.3 stay selectable
        for cheaper cached reads and long-context runs respectively."""
        from discord_grok.cogs.grok.command_options import (
            CHAT_MODEL_INDEX,
            DEFAULT_CHAT_MODEL_ID,
        )

        assert DEFAULT_CHAT_MODEL_ID == "grok-4.6"
        assert "grok-4.5" in CHAT_MODEL_INDEX
        assert "grok-4.3" in CHAT_MODEL_INDEX

    def test_model_markdown_lines_match_visible_models(self):
        """README model-list helper should reflect the visible slash-command models."""
        from discord_grok.cogs.grok.command_options import (
            generate_model_markdown_lines,
            iter_slash_command_models,
        )

        assert generate_model_markdown_lines() == [
            f"- `{entry.model_id}` — {entry.display_name} ({entry.pricing_class})"
            for entry in iter_slash_command_models()
        ]

    def test_chat_exposes_supported_grok_chat_options(self, cog):
        """Chat command should expose the supported Grok chat surface."""
        chat_cmd = next(cmd for cmd in cog.grok.walk_commands() if cmd.name == "chat")
        option_names = {opt.name for opt in chat_cmd.options}

        assert option_names == {
            "prompt",
            "system_prompt",
            "model",
            "attachment",
            "max_tokens",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "reasoning_effort",
            "agent_count",
            "web_search",
            "x_search",
            "code_execution",
            "collections_search",
            "mcp",
            "x_search_images",
            "x_search_videos",
            "x_search_date_range",
            "web_search_images",
        }

    def test_image_model_choices_match_grok_image_models(self, cog):
        """Image command model choices should match GROK_IMAGE_MODELS."""
        from discord_grok.cogs.grok.tooling import GROK_IMAGE_MODELS

        image_cmd = next(cmd for cmd in cog.grok_media.walk_commands() if cmd.name == "image")
        model_option = next(opt for opt in image_cmd.options if opt.name == "model")
        choice_values = sorted(choice.value for choice in model_option.choices)
        assert choice_values == sorted(GROK_IMAGE_MODELS)

    def test_image_aspect_ratios_match_sdk(self, cog):
        """Image command aspect ratios should match xai-sdk ImageAspectRatio."""
        from xai_sdk.image import ImageAspectRatio

        image_cmd = next(cmd for cmd in cog.grok_media.walk_commands() if cmd.name == "image")
        ar_option = next(opt for opt in image_cmd.options if opt.name == "aspect_ratio")
        choice_values = sorted(choice.value for choice in ar_option.choices)
        assert choice_values == sorted(ImageAspectRatio.__args__)

    def test_image_resolution_choices_match_sdk(self, cog):
        """Image command resolution choices should match xai-sdk ImageResolution."""
        from xai_sdk.image import ImageResolution

        image_cmd = next(cmd for cmd in cog.grok_media.walk_commands() if cmd.name == "image")
        res_option = next(opt for opt in image_cmd.options if opt.name == "resolution")
        choice_values = sorted(choice.value for choice in res_option.choices)
        assert choice_values == sorted(ImageResolution.__args__)

    def test_image_has_attachment_option(self, cog):
        """Image command should have an optional attachment parameter."""
        image_cmd = next(cmd for cmd in cog.grok_media.walk_commands() if cmd.name == "image")
        att_option = next((opt for opt in image_cmd.options if opt.name == "attachment"), None)
        assert att_option is not None
        assert att_option.required is False

    def test_video_aspect_ratios_match_sdk(self, cog):
        """Video command aspect ratios should match xai-sdk VideoAspectRatio."""
        from xai_sdk.video import VideoAspectRatio

        video_cmd = next(cmd for cmd in cog.grok_media.walk_commands() if cmd.name == "video")
        ar_option = next(opt for opt in video_cmd.options if opt.name == "aspect_ratio")
        choice_values = sorted(choice.value for choice in ar_option.choices)
        assert choice_values == sorted(VideoAspectRatio.__args__)

    def test_video_has_attachment_option(self, cog):
        """Video command should have an optional attachment parameter."""
        video_cmd = next(cmd for cmd in cog.grok_media.walk_commands() if cmd.name == "video")
        att_option = next((opt for opt in video_cmd.options if opt.name == "attachment"), None)
        assert att_option is not None
        assert att_option.required is False

    def test_image_resolution_choices_are_all_priced(self, cog):
        """Grok Imagine bills per output resolution, so every resolution the image
        command offers needs a rate for every image model. A missing rate makes the
        cost embed silently fall back to an unrelated tier's price.

        grok-imagine-image-2.0 prices on resolution AND quality, so its rates are
        keyed ``"<resolution>/<quality>"`` — a bare resolution counts as priced when
        any quality tier for it exists.
        """
        # Read the maps through tooling: tests that reload config.pricing under an
        # XAI_PRICING_PATH override leave a custom module in sys.modules.
        from discord_grok.cogs.grok.tooling import GROK_IMAGE_MODELS, IMAGE_PRICING

        cmd = next(c for c in cog.grok_media.walk_commands() if c.name == "image")
        res_option = next(opt for opt in cmd.options if opt.name == "resolution")
        for model in GROK_IMAGE_MODELS:
            rates = IMAGE_PRICING[model]
            for choice in res_option.choices:
                priced = choice.value in rates or any(
                    key.startswith(f"{choice.value}/") for key in rates
                )
                assert priced, f"image: {model} has no {choice.value} rate"

    def test_every_image_quality_tier_is_priced(self, cog):
        """Both quality choices must be priced at both resolutions for the two-axis
        model, or a request bills at the fail-high tier instead of its own rate."""
        from discord_grok.cogs.grok.tooling import IMAGE_PRICING

        cmd = next(c for c in cog.grok_media.walk_commands() if c.name == "image")
        res_option = next(opt for opt in cmd.options if opt.name == "resolution")
        quality_option = next(opt for opt in cmd.options if opt.name == "quality")
        rates = IMAGE_PRICING["grok-imagine-image-2.0"]
        for res in (c.value for c in res_option.choices):
            for quality in (c.value for c in quality_option.choices):
                assert f"{res}/{quality}" in rates, f"unpriced tier {res}/{quality}"

    def test_video_resolution_choices_are_priced_or_rejected(self, cog):
        """1080p exists only on Video 1.5. Every offered (model, resolution) pair must
        therefore be either priced or refused before the request, never billed at the
        unknown-model fallback."""
        from discord_grok.cogs.grok.tooling import GROK_VIDEO_MODELS, VIDEO_PRICING
        from discord_grok.cogs.grok.video import _validate_video_resolution

        cmd = next(c for c in cog.grok_media.walk_commands() if c.name == "video")
        res_option = next(opt for opt in cmd.options if opt.name == "resolution")
        for model in GROK_VIDEO_MODELS:
            for choice in res_option.choices:
                if choice.value in VIDEO_PRICING[model]:
                    assert _validate_video_resolution(model, choice.value) is None
                else:
                    assert _validate_video_resolution(model, choice.value) is not None, (
                        f"video: {model} offers unpriced {choice.value} without rejecting it"
                    )

    def test_1080p_is_offered_and_priced_on_video_1_5(self, cog):
        """The 1080p rate existed in pricing.yaml while the menu offered only 720p/480p,
        so the tier was unreachable."""
        from discord_grok.cogs.grok.tooling import VIDEO_PRICING

        cmd = next(c for c in cog.grok_media.walk_commands() if c.name == "video")
        res_option = next(opt for opt in cmd.options if opt.name == "resolution")
        assert any(choice.value == "1080p" for choice in res_option.choices)
        assert VIDEO_PRICING["grok-imagine-video-1.5-preview"]["1080p"] == 0.25

    def test_media_resolution_defaults_match_the_assumed_pricing_tier(self, cog):
        """``calculate_image_cost``/``calculate_video_cost`` assume these resolutions
        when a caller passes none, so a command default that drifts away from them
        would bill the wrong tier — exactly how the 720p default came to be estimated
        at the 480p rate."""
        import inspect

        from discord_grok.cogs.grok.tooling import (
            DEFAULT_IMAGE_RESOLUTION,
            DEFAULT_VIDEO_RESOLUTION,
        )

        video_cmd = next(c for c in cog.grok_media.walk_commands() if c.name == "video")
        video_res = next(opt for opt in video_cmd.options if opt.name == "resolution")
        video_default = inspect.signature(cog.video.callback).parameters["resolution"].default
        assert video_default == DEFAULT_VIDEO_RESOLUTION
        assert f"default: {DEFAULT_VIDEO_RESOLUTION}" in video_res.description

        # The image command leaves resolution unset so the API applies its own
        # default; the option documents which tier that is.
        image_cmd = next(c for c in cog.grok_media.walk_commands() if c.name == "image")
        image_res = next(opt for opt in image_cmd.options if opt.name == "resolution")
        assert inspect.signature(cog.image.callback).parameters["resolution"].default is None
        assert f"default: {DEFAULT_IMAGE_RESOLUTION}" in image_res.description

    def test_tts_voice_choices_match_tts_voices(self, cog):
        """TTS command voice choices should match TTS_VOICES."""
        from discord_grok.cogs.grok.tooling import TTS_VOICES

        tts_cmd = next(cmd for cmd in cog.grok_tools.walk_commands() if cmd.name == "tts")
        voice_option = next(opt for opt in tts_cmd.options if opt.name == "voice")
        choice_values = sorted(choice.value for choice in voice_option.choices)
        assert choice_values == sorted(TTS_VOICES)


class TestTTSCommand:
    """Tests for the /grok-tools tts command."""

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
        image_cmd = next(cmd for cmd in cog.grok_media.walk_commands() if cmd.name == "image")
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

        from discord_grok.cogs.grok.state import _extract_daily_total

        key = (mock_discord_context.author.id, date.today().isoformat())
        assert abs(_extract_daily_total(cog.daily_costs[key]) - expected_cost) < 1e-9

    async def test_image_cost_prefers_sdk_reported_cost_usd(self, cog, mock_discord_context):
        """When the SDK response carries cost_usd, that value should be used over YAML pricing."""
        cog.client.image.sample.return_value.cost_usd = 0.123

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

        from datetime import date

        from discord_grok.cogs.grok.state import _extract_daily_total

        key = (mock_discord_context.author.id, date.today().isoformat())
        assert abs(_extract_daily_total(cog.daily_costs[key]) - 0.123) < 1e-9

    async def test_image_batch_cost_mixes_sdk_and_yaml_per_result(self, cog, mock_discord_context):
        """Mixed Some/None cost_usd across batch should sum SDK values + YAML fallback per missing."""
        from discord_grok.cogs.grok.tooling import calculate_image_cost

        reported = MagicMock()
        reported.url = "https://example.com/r1.png"
        reported.base64 = None
        reported.cost_usd = 0.05

        unreported = MagicMock()
        unreported.url = "https://example.com/r2.png"
        unreported.base64 = None
        unreported.cost_usd = None

        cog.client.image.sample_batch.return_value = [reported, unreported]

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

        from datetime import date

        from discord_grok.cogs.grok.state import _extract_daily_total

        expected = 0.05 + calculate_image_cost("grok-imagine-image")
        key = (mock_discord_context.author.id, date.today().isoformat())
        assert abs(_extract_daily_total(cog.daily_costs[key]) - expected) < 1e-9

    async def test_image_yaml_fallback_cost_uses_requested_resolution(
        self, cog, mock_discord_context
    ):
        """2k output costs more than 1k on the quality tier, and an unset resolution
        bills at the 1k rate the command documents as the default."""
        from datetime import date

        from discord_grok.cogs.grok.state import _extract_daily_total

        key = (mock_discord_context.author.id, date.today().isoformat())
        for resolution, expected in ((None, 0.05), ("1k", 0.05), ("2k", 0.07)):
            cog.daily_costs.clear()
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
                    model="grok-imagine-image-quality",
                    resolution=resolution,
                    count=1,
                )

            assert abs(_extract_daily_total(cog.daily_costs[key]) - expected) < 1e-9

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
    """Integration tests for the /grok-media video command."""

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

    async def test_video_cost_prefers_sdk_reported_cost_usd(self, cog, mock_discord_context):
        """When the SDK response carries cost_usd, that value should be used over YAML pricing."""
        cog.client.video.generate.return_value.cost_usd = 0.42

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

        from datetime import date

        from discord_grok.cogs.grok.state import _extract_daily_total

        key = (mock_discord_context.author.id, date.today().isoformat())
        assert abs(_extract_daily_total(cog.daily_costs[key]) - 0.42) < 1e-9

    async def test_video_yaml_fallback_cost_uses_requested_resolution(
        self, cog, mock_discord_context
    ):
        """The YAML fallback must bill the resolution actually requested: on
        grok-imagine-video-1.5-preview the 720p default costs 1.75x what 480p does."""
        from datetime import date

        from discord_grok.cogs.grok.state import _extract_daily_total

        key = (mock_discord_context.author.id, date.today().isoformat())
        for resolution, expected in (("720p", 5 * 0.14), ("480p", 5 * 0.08)):
            cog.daily_costs.clear()
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
                    duration=5,
                    resolution=resolution,
                )

            assert abs(_extract_daily_total(cog.daily_costs[key]) - expected) < 1e-9

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
