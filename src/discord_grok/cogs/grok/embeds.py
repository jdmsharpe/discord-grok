from discord import Colour, Embed

from .models import CitationInfo
from .tooling import (
    CHUNK_TEXT_SIZE,
    TOOL_USAGE_DISPLAY_NAMES,
    calculate_tool_cost,
    chunk_text,
    truncate_text,
)

GROK_BLACK = Colour(0x000000)
REASONING_TRUNCATION_SUFFIX = "\n\n... [reasoning truncated]"


def append_reasoning_embeds(embeds: list[Embed], reasoning_text: str) -> None:
    """Append reasoning text as a spoilered Discord embed."""
    if not reasoning_text:
        return
    if len(reasoning_text) > CHUNK_TEXT_SIZE:
        reasoning_text = (
            reasoning_text[: CHUNK_TEXT_SIZE - len(REASONING_TRUNCATION_SUFFIX)]
            + REASONING_TRUNCATION_SUFFIX
        )
    embeds.append(
        Embed(
            title="Reasoning",
            description=f"||{reasoning_text}||",
            color=Colour.light_grey(),
        )
    )


def append_response_embeds(embeds: list[Embed], response_text: str) -> None:
    """Append response text as Discord embeds, handling chunking for long responses."""
    for index, chunk in enumerate(chunk_text(response_text), start=1):
        embeds.append(
            Embed(
                title="Response" + (f" (Part {index})" if index > 1 else ""),
                description=chunk,
                color=GROK_BLACK,
            )
        )


def append_sources_embed(embeds: list[Embed], citations: list[CitationInfo]) -> None:
    """Append a compact sources embed for tool-backed responses, grouped by type."""
    if not citations or len(embeds) >= 10:
        return

    web: list[CitationInfo] = []
    x: list[CitationInfo] = []
    collections: list[CitationInfo] = []
    for cit in citations:
        if cit["source"] == "x":
            x.append(cit)
        elif cit["source"] == "collections":
            collections.append(cit)
        else:
            web.append(cit)

    parts: list[str] = []

    def _format_link_group(heading: str | None, items: list[CitationInfo], limit: int = 8) -> None:
        if not items:
            return
        lines: list[str] = []
        if heading:
            lines.append(f"**{heading}**")
        for index, cit in enumerate(items[:limit], start=1):
            url = cit["url"]
            if url.startswith("http://") or url.startswith("https://"):
                title = truncate_text(url.removeprefix("https://").removeprefix("http://"), 120)
                lines.append(f"{index}. [{title}]({url})")
            else:
                lines.append(f"{index}. `{truncate_text(url, 300)}`")
        parts.append("\n".join(lines))

    has_multiple_types = sum(bool(g) for g in (web, x, collections)) > 1
    _format_link_group("Web" if has_multiple_types else None, web)
    _format_link_group("X Posts" if has_multiple_types else None, x)
    _format_link_group("Collections" if has_multiple_types else None, collections)

    description = "\n\n".join(parts)
    if len(description) > 4000:
        description = truncate_text(description, 3990)

    embeds.append(Embed(title="Sources", description=description, color=GROK_BLACK))


def append_pricing_embed(
    embeds: list[Embed],
    cost: float,
    input_tokens: int,
    output_tokens: int,
    daily_cost: float,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    image_tokens: int = 0,
    tool_usage: dict[str, int] | None = None,
) -> None:
    """Append a compact pricing embed showing cost and token usage."""
    tool_cost = calculate_tool_cost(tool_usage) if tool_usage else 0.0
    in_qualifiers = []
    if cached_tokens > 0:
        in_qualifiers.append(f"{cached_tokens:,} cached")
    if image_tokens > 0:
        in_qualifiers.append(f"{image_tokens:,} image")
    token_info = f"{input_tokens:,} tokens in"
    if in_qualifiers:
        token_info += f" ({', '.join(in_qualifiers)})"
    token_info += f" / {output_tokens:,} tokens out"
    if reasoning_tokens > 0:
        token_info += f" ({reasoning_tokens:,} reasoning)"
    description = f"${cost:.4f} · {token_info} · daily ${daily_cost:.2f}"
    if tool_usage:
        tool_parts = []
        for key, count in tool_usage.items():
            name = TOOL_USAGE_DISPLAY_NAMES.get(
                key,
                key.replace("SERVER_SIDE_TOOL_", "").replace("_", " ").title(),
            )
            tool_parts.append(f"{name} ×{count}")
        if tool_cost > 0:
            tool_parts.append(f"tool cost ${tool_cost:.4f}")
        description += "\n" + " · ".join(tool_parts)
    embeds.append(Embed(description=description, color=GROK_BLACK))


def append_generation_pricing_embed(
    embeds: list[Embed],
    cost: float,
    daily_cost: float,
) -> None:
    """Append a compact pricing embed for image/video generation."""
    embeds.append(Embed(description=f"${cost:.4f} · daily ${daily_cost:.2f}", color=GROK_BLACK))


__all__ = [
    "GROK_BLACK",
    "append_generation_pricing_embed",
    "append_pricing_embed",
    "append_reasoning_embeds",
    "append_response_embeds",
    "append_sources_embed",
]
