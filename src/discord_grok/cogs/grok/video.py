from __future__ import annotations

import io
from typing import cast

from discord import ApplicationContext, Attachment, Colour, Embed, File
from xai_sdk.video import VideoAspectRatio, VideoResolution

from .embed_delivery import send_embed_batches
from .embeds import GROK_BLACK, append_generation_pricing_embed
from .tooling import GROK_VIDEO_MODELS, calculate_video_cost, format_xai_error, truncate_text


async def run_video_command(
    cog,
    *,
    ctx: ApplicationContext,
    prompt: str,
    aspect_ratio: str,
    duration: int,
    resolution: str,
    attachment: Attachment | None,
) -> None:
    """Generate a video from text or an image using Grok Imagine Video."""
    await ctx.defer()

    try:
        is_image_to_video = attachment is not None
        mode = "Image-to-Video" if is_image_to_video else "Text-to-Video"
        cog.logger.info("Starting video generation with grok-imagine-video (mode=%s)", mode)

        generate_kwargs = {
            "prompt": prompt,
            "model": GROK_VIDEO_MODELS[0],
            "aspect_ratio": cast(VideoAspectRatio, aspect_ratio),
            "duration": duration,
            "resolution": cast(VideoResolution, resolution),
        }
        if is_image_to_video:
            generate_kwargs["image_url"] = str(attachment.url)

        client = cog._get_client()
        result = await client.video.generate(**generate_kwargs)
        if not result.url:
            raise Exception("No video URL returned from the API.")

        session = await cog._get_http_session()
        async with session.get(result.url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download video: HTTP {response.status}")
            video_bytes = await response.read()

        video_cost = calculate_video_cost(duration)
        daily_cost = cog._track_daily_cost(ctx.author.id, video_cost)

        cog.logger.info(
            "COST | command=video | user=%s | duration=%ds | cost=$%.4f | daily=$%.4f",
            ctx.author.id,
            duration,
            video_cost,
            daily_cost,
        )

        data = io.BytesIO(video_bytes)
        description = f"**Prompt:** {truncate_text(prompt, 2000)}\n"
        description += f"**Mode:** {mode}\n"
        description += f"**Aspect Ratio:** {aspect_ratio}\n"
        description += f"**Duration:** {duration}s\n"
        description += f"**Resolution:** {resolution}\n"

        embeds = [
            Embed(
                title="Video Generation",
                description=description,
                color=GROK_BLACK,
            )
        ]
        if cog.show_cost_embeds:
            append_generation_pricing_embed(embeds, video_cost, daily_cost)
        await send_embed_batches(
            ctx.send_followup,
            embeds=embeds,
            file=File(data, "video.mp4"),
            logger=cog.logger,
        )
        cog.logger.info("Successfully generated and sent video")

    except Exception as error:
        description = format_xai_error(error)
        cog.logger.error("Video generation failed: %s", description, exc_info=True)
        await send_embed_batches(
            ctx.send_followup,
            embed=Embed(title="Error", description=description, color=Colour.red()),
            logger=cog.logger,
        )


__all__ = ["run_video_command"]
