from __future__ import annotations

import re
from typing import Any, Literal

from .models import CitationInfo, ToolInfo

_CITATION_MARKER_RE = re.compile(r"\[\[\d+\]\]\([^)]+\)")


def classify_citation_url(url: str) -> Literal["web", "x", "collections"]:
    """Classify a citation URL into its source type."""
    if url.startswith("collections://"):
        return "collections"
    if url.startswith("https://x.com/") or url.startswith("https://twitter.com/"):
        return "x"
    return "web"


def extract_tool_info(response_json: dict[str, Any]) -> ToolInfo:
    """Extract structured citation data from a Responses API JSON response."""
    citations: list[CitationInfo] = []
    seen_urls: set[str] = set()

    for output_item in response_json.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_part in output_item.get("content", []):
            if not isinstance(content_part, dict):
                continue
            for annotation in content_part.get("annotations", []):
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("type") != "url_citation":
                    continue
                url = str(annotation.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                citations.append({"url": url, "source": classify_citation_url(url)})

    return {"citations": citations}


def extract_response_text(response_json: dict[str, Any]) -> tuple[str, str]:
    """Extract response text and reasoning text from a Responses API response."""
    response_text = ""
    reasoning_text = ""
    for output_item in response_json.get("output", []):
        if not isinstance(output_item, dict):
            continue
        if output_item.get("type") == "reasoning":
            for part in output_item.get("summary", []):
                if isinstance(part, dict) and part.get("type") == "summary_text":
                    reasoning_text += part.get("text", "")
        elif output_item.get("role") == "assistant":
            for content_part in output_item.get("content", []):
                if isinstance(content_part, dict) and content_part.get("type") == "output_text":
                    response_text += content_part.get("text", "")
    response_text = _CITATION_MARKER_RE.sub("", response_text).strip()
    return response_text or "No response.", reasoning_text


# usage.cost_in_usd_ticks is denominated in 1e-10 USD (usage.proto: "Full price paid").
_USD_TICKS_PER_DOLLAR = 10_000_000_000

# usage.server_side_tool_usage_details counters -> pricing.yaml `tools` keys.
_TOOL_USAGE_DETAIL_KEYS: dict[str, str] = {
    "web_search_calls": "SERVER_SIDE_TOOL_WEB_SEARCH",
    "x_search_calls": "SERVER_SIDE_TOOL_X_SEARCH",
    "code_interpreter_calls": "SERVER_SIDE_TOOL_CODE_INTERPRETER",
    "file_search_calls": "SERVER_SIDE_TOOL_FILE_SEARCH",
    "mcp_calls": "SERVER_SIDE_TOOL_MCP",
    "document_search_calls": "SERVER_SIDE_TOOL_DOCUMENT_SEARCH",
    "image_generation_calls": "SERVER_SIDE_TOOL_IMAGE_GENERATION",
}


def extract_usage(response_json: dict[str, Any]) -> dict[str, Any]:
    """Extract token usage from a Responses API response.

    ``cost_usd`` is the price xAI reports for the request
    (``usage.cost_in_usd_ticks``, token and server-side tool charges included)
    or None when absent.
    """
    usage = response_json.get("usage", {})
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = (
        usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    )
    ticks = usage.get("cost_in_usd_ticks")
    cost_usd = (
        ticks / _USD_TICKS_PER_DOLLAR
        if isinstance(ticks, int | float) and not isinstance(ticks, bool) and ticks >= 0
        else None
    )
    return {
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens") or 0,
        "reasoning_tokens": output_details.get("reasoning_tokens", 0) or 0,
        "cached_tokens": input_details.get("cached_tokens", 0) or 0,
        "image_tokens": input_details.get("image_tokens", 0) or 0,
        "cost_usd": cost_usd,
    }


def extract_tool_usage(response_json: dict[str, Any]) -> dict[str, int]:
    """Server-side tool call counts keyed like pricing.yaml's ``tools`` rows.

    Current responses report the counters under
    ``usage.server_side_tool_usage_details`` (``x_search_calls`` and the other
    ``*_calls`` fields); a top-level ``server_side_tool_usage`` map is the earlier
    shape and is returned as-is when present.
    """
    legacy = response_json.get("server_side_tool_usage")
    if isinstance(legacy, dict) and legacy:
        return legacy
    details = (response_json.get("usage") or {}).get("server_side_tool_usage_details") or {}
    counts: dict[str, int] = {}
    for detail_name, usage_key in _TOOL_USAGE_DETAIL_KEYS.items():
        count = details.get(detail_name)
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            counts[usage_key] = count
    return counts


__all__ = [
    "extract_response_text",
    "extract_tool_info",
    "extract_tool_usage",
    "extract_usage",
]
