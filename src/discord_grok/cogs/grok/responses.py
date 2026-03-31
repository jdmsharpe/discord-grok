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


def extract_usage(response_json: dict[str, Any]) -> dict[str, int]:
    """Extract token usage from a Responses API response."""
    usage = response_json.get("usage", {})
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = (
        usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    )
    return {
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens") or 0,
        "reasoning_tokens": output_details.get("reasoning_tokens", 0) or 0,
        "cached_tokens": input_details.get("cached_tokens", 0) or 0,
        "image_tokens": input_details.get("image_tokens", 0) or 0,
    }


__all__ = [
    "extract_response_text",
    "extract_tool_info",
    "extract_usage",
]
