from __future__ import annotations

import io

from discord import ApplicationContext, Colour, Embed, File

from .client import TTS_MAX_CHARS
from .embed_delivery import send_embed_batches
from .embeds import GROK_BLACK, append_generation_pricing_embed
from .tooling import calculate_tts_cost, format_xai_error, truncate_text


async def run_tts_command(
    cog,
    *,
    ctx: ApplicationContext,
    text: str,
    voice: str,
    language: str,
    output_format: str,
    sample_rate: int | None,
    bit_rate: int | None,
) -> None:
    """Convert text to speech using the xAI TTS API."""
    await ctx.defer()

    try:
        if len(text) > TTS_MAX_CHARS:
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="Error",
                    description=(
                        f"Text exceeds the {TTS_MAX_CHARS:,} character limit "
                        f"({len(text):,} characters provided)."
                    ),
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
            return

        cog.logger.info(
            "Generating TTS with voice=%s, language=%s, format=%s",
            voice,
            language,
            output_format,
        )
        audio_bytes = await cog._generate_tts(
            text,
            voice,
            language,
            output_format,
            sample_rate,
            bit_rate,
        )

        tts_cost = calculate_tts_cost(len(text))
        daily_cost = cog._track_daily_cost(ctx.author.id, tts_cost)

        cog.logger.info(
            "COST | command=tts | user=%s | voice=%s | chars=%d | cost=$%.4f | daily=$%.4f",
            ctx.author.id,
            voice,
            len(text),
            tts_cost,
            daily_cost,
        )

        description = f"**Text:** {truncate_text(text, 2000)}\n"
        description += f"**Voice:** {voice}\n"
        description += f"**Language:** {language}\n"
        output_description = output_format
        if sample_rate is not None:
            output_description += f" @ {sample_rate:,} Hz"
        if bit_rate is not None and output_format == "mp3":
            output_description += f" / {bit_rate // 1000} kbps"
        description += f"**Format:** {output_description}\n"

        embeds = [
            Embed(
                title="Text-to-Speech Generation",
                description=description,
                color=GROK_BLACK,
            )
        ]
        if cog.show_cost_embeds:
            append_generation_pricing_embed(embeds, tts_cost, daily_cost)
        extension = "ulaw" if output_format == "mulaw" else output_format
        await send_embed_batches(
            ctx.send_followup,
            embeds=embeds,
            file=File(io.BytesIO(audio_bytes), f"speech.{extension}"),
            logger=cog.logger,
        )
        cog.logger.info("Successfully generated and sent TTS audio")

    except Exception as error:
        description = format_xai_error(error)
        cog.logger.error("TTS generation failed: %s", description, exc_info=True)
        await send_embed_batches(
            ctx.send_followup,
            embed=Embed(title="Error", description=description, color=Colour.red()),
            logger=cog.logger,
        )


__all__ = ["run_tts_command"]
