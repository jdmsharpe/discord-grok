from __future__ import annotations

import base64
import io
from typing import Any, cast

from discord import ApplicationContext, Attachment, Colour, Embed, File
from xai_sdk.image import ImageAspectRatio, ImageResolution

from .embed_delivery import send_embed_batches
from .embeds import GROK_BLACK, append_generation_pricing_embed
from .tooling import calculate_image_cost, format_xai_error, truncate_text


async def run_image_command(
    cog,
    *,
    ctx: ApplicationContext,
    prompt: str,
    model: str,
    aspect_ratio: str,
    resolution: str | None,
    count: int,
    attachment: Attachment | None,
) -> None:
    """Generate or edit images using Grok Imagine."""
    await ctx.defer()

    try:
        is_editing = attachment is not None
        mode = "Image Editing" if is_editing else "Image Generation"

        if is_editing and count > 1:
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="Error",
                    description=("Multi-image generation is not supported in Image Editing mode."),
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
            return

        cog.logger.info("Generating image with model %s (mode=%s, count=%d)", model, mode, count)

        sample_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": cast(ImageAspectRatio, aspect_ratio),
        }
        if resolution is not None:
            sample_kwargs["resolution"] = cast(ImageResolution, resolution)
        if is_editing:
            sample_kwargs["image_url"] = str(attachment.url)

        client = cog._get_client()
        if count == 1:
            result = await client.image.sample(**sample_kwargs)
            results = [result]
        else:
            results = await client.image.sample_batch(n=count, **sample_kwargs)

        image_cost = calculate_image_cost(model) * len(results)
        daily_cost = cog._track_daily_cost(ctx.author.id, image_cost)

        cog.logger.info(
            "COST | command=image | user=%s | model=%s | count=%d | cost=$%.4f | daily=$%.4f",
            ctx.author.id,
            model,
            len(results),
            image_cost,
            daily_cost,
        )

        session = await cog._get_http_session()
        files: list[File] = []
        for index, image_result in enumerate(results):
            if image_result.url:
                async with session.get(image_result.url) as response:
                    if response.status != 200:
                        raise Exception(
                            f"Failed to download image {index + 1}: HTTP {response.status}"
                        )
                    data = io.BytesIO(await response.read())
            elif image_result.base64:
                data = io.BytesIO(base64.b64decode(image_result.base64))
            else:
                raise Exception(f"No image data returned for image {index + 1}.")
            filename = f"image_{index + 1}.png" if len(results) > 1 else "image.png"
            files.append(File(data, filename))

        description = f"**Prompt:** {truncate_text(prompt, 2000)}\n"
        description += f"**Model:** {model}\n"
        description += f"**Mode:** {mode}\n"
        if count > 1:
            description += f"**Count:** {count}\n"
        description += f"**Aspect Ratio:** {aspect_ratio}\n"
        if resolution is not None:
            description += f"**Resolution:** {resolution}\n"

        embed = Embed(
            title=mode,
            description=description,
            color=GROK_BLACK,
        )
        embed.set_image(url=f"attachment://{files[0].filename}")
        embeds = [embed]
        if cog.show_cost_embeds:
            append_generation_pricing_embed(embeds, image_cost, daily_cost)
        await send_embed_batches(ctx.send_followup, embeds=embeds, files=files, logger=cog.logger)
        cog.logger.info("Successfully generated and sent %d image(s)", len(results))

    except Exception as error:
        description = format_xai_error(error)
        cog.logger.error("Image generation failed: %s", description, exc_info=True)
        await send_embed_batches(
            ctx.send_followup,
            embed=Embed(title="Error", description=description, color=Colour.red()),
            logger=cog.logger,
        )


__all__ = ["run_image_command"]
