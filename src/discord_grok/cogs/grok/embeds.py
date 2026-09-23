from collections.abc import Iterable
from urllib.parse import urlparse

from discord import Colour, Embed

from ...cost_line import count_label, format_cost_line
from .models import CitationInfo
from .tooling import (
    CHUNK_TEXT_SIZE,
    TOOL_USAGE_DETAIL_LABELS,
    chunk_text,
    truncate_text,
)

GROK_BLACK = Colour(0x000000)
REASONING_TRUNCATION_SUFFIX = "\n\n... [reasoning truncated]"


def _fit_markdown_sections(
    sections: list[tuple[str | None, list[str]]],
    max_length: int = 4000,
) -> str:
    """Fit complete Markdown entries without slicing through links."""

    rendered_sections: list[str] = []
    for heading, entries in sections:
        accepted: list[str] = []
        for entry in entries:
            body = "\n".join([*accepted, entry])
            rendered = f"{heading}\n{body}" if heading else body
            candidate = "\n\n".join([*rendered_sections, rendered])
            if len(candidate) > max_length:
                break
            accepted.append(entry)
        if accepted:
            body = "\n".join(accepted)
            rendered_sections.append(f"{heading}\n{body}" if heading else body)
    return "\n\n".join(rendered_sections)


def _source_hostname(url: str) -> str | None:
    host = urlparse(url).hostname
    return host.removeprefix("www.") if host else None


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

    sections: list[tuple[str | None, list[str]]] = []

    def _format_link_group(heading: str | None, items: list[CitationInfo], limit: int = 8) -> None:
        if not items:
            return
        lines: list[str] = []
        for index, cit in enumerate(items[:limit], start=1):
            url = cit["url"]
            if url.startswith("http://") or url.startswith("https://"):
                title = _source_hostname(url) or f"Source {index}"
                lines.append(f"{index}. [{title}]({url})")
            else:
                lines.append(f"{index}. `{truncate_text(url, 300)}`")
        sections.append((f"**{heading}**" if heading else None, lines))

    has_multiple_types = sum(bool(g) for g in (web, x, collections)) > 1
    _format_link_group("Web" if has_multiple_types else None, web)
    _format_link_group("X Posts" if has_multiple_types else None, x)
    _format_link_group("Collections" if has_multiple_types else None, collections)

    description = _fit_markdown_sections(sections)
    if not description:
        return

    embeds.append(Embed(title="Sources", description=description, color=GROK_BLACK))


def _tool_details(tool_usage: dict[str, int]) -> list[str]:
    """Return the cost-line tool counts, such as ``["2 searches", "1 code run"]``."""
    counts: dict[tuple[str, str | None], int] = {}
    for key, labels in TOOL_USAGE_DETAIL_LABELS.items():
        if tool_usage.get(key, 0) > 0:
            counts[labels] = counts.get(labels, 0) + tool_usage[key]
    for key, count in tool_usage.items():
        if key not in TOOL_USAGE_DETAIL_LABELS and count > 0:
            name = key.removeprefix("SERVER_SIDE_TOOL_").replace("_", " ").lower()
            labels = (f"{name} call", None)
            counts[labels] = counts.get(labels, 0) + count
    return [count_label(count, singular, plural) for (singular, plural), count in counts.items()]


def append_pricing_embed(
    embeds: list[Embed],
    cost: float,
    input_tokens: int,
    output_tokens: int,
    daily_cost: float,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    tool_usage: dict[str, int] | None = None,
) -> None:
    """Append the one-line cost embed for a chat response.

    ``cost`` is the whole request cost, tool charges included. The token counts are
    xAI's usage fields: ``input_tokens`` includes ``cached_tokens`` and
    ``output_tokens`` includes ``reasoning_tokens``.
    """
    line = format_cost_line(
        cost,
        daily_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        thinking_tokens=reasoning_tokens,
        details=_tool_details(tool_usage or {}),
    )
    embeds.append(Embed(description=line, color=GROK_BLACK))


def append_generation_pricing_embed(
    embeds: list[Embed],
    cost: float,
    daily_cost: float,
    *,
    details: Iterable[str],
) -> None:
    """Append the one-line cost embed for an image, video, or speech command."""
    line = format_cost_line(cost, daily_cost, details=details)
    embeds.append(Embed(description=line, color=GROK_BLACK))


__all__ = [
    "GROK_BLACK",
    "append_generation_pricing_embed",
    "append_pricing_embed",
    "append_reasoning_embeds",
    "append_response_embeds",
    "append_sources_embed",
]
