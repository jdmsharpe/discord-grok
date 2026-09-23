from unittest.mock import MagicMock

from discord import Colour, Embed


class TestAppendPricingEmbed:
    """Tests for the append_pricing_embed helper."""

    def test_append_pricing_embed(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50)
        assert len(embeds) == 1
        assert embeds[0].description == "$0.0500 · 1k in / 500 out · $1.50 today"
        assert embeds[0].colour == Colour(0)

    def test_append_pricing_embed_shows_reasoning_as_part_of_output(self):
        """xAI's output_tokens includes reasoning_tokens; the line shows it unchanged."""
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 700, 1.50, reasoning_tokens=200)
        assert len(embeds) == 1
        assert embeds[0].description == "$0.0500 · 1k in / 700 out (200 thinking) · $1.50 today"

    def test_append_pricing_embed_hides_zero_reasoning_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, reasoning_tokens=0)
        assert embeds[0].description == "$0.0500 · 1k in / 500 out · $1.50 today"

    def test_append_pricing_embed_with_cached_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, cached_tokens=300)
        assert embeds[0].description == "$0.0500 · 1k in (300 cached) / 500 out · $1.50 today"

    def test_append_pricing_embed_hides_zero_cached_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, cached_tokens=0)
        assert embeds[0].description == "$0.0500 · 1k in / 500 out · $1.50 today"

    def test_append_pricing_embed_full_line(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(
            embeds,
            0.0014,
            1252,
            107,
            0.004,
            reasoning_tokens=102,
            cached_tokens=1152,
            tool_usage={"SERVER_SIDE_TOOL_WEB_SEARCH": 1},
        )
        assert embeds[0].description == (
            "$0.0014 · 1.3k in (1.2k cached) / 107 out (102 thinking) · 1 search · <$0.01 today"
        )

    def test_append_pricing_embed_with_tool_usage(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        tool_usage = {"SERVER_SIDE_TOOL_WEB_SEARCH": 3, "SERVER_SIDE_TOOL_X_SEARCH": 2}
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, tool_usage=tool_usage)
        assert embeds[0].description == (
            "$0.0500 · 1k in / 500 out · 3 searches · 2 X searches · $1.50 today"
        )

    def test_append_pricing_embed_orders_and_merges_tool_counts(self):
        """Tools appear in the fleet order, keys that share a label are summed, and an
        unlisted key is shown as "<name> call"."""
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        tool_usage = {
            "SERVER_SIDE_TOOL_NEW_TOOL": 2,
            "SERVER_SIDE_TOOL_VIEW_X_VIDEO": 1,
            "SERVER_SIDE_TOOL_MCP": 1,
            "SERVER_SIDE_TOOL_FILE_SEARCH": 2,
            "SERVER_SIDE_TOOL_CODE_INTERPRETER": 1,
            "SERVER_SIDE_TOOL_CODE_EXECUTION": 1,
            "SERVER_SIDE_TOOL_X_SEARCH": 1,
            "SERVER_SIDE_TOOL_WEB_SEARCH": 1,
            "SERVER_SIDE_TOOL_ATTACHMENT_SEARCH": 0,
        }
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, tool_usage=tool_usage)
        assert embeds[0].description == (
            "$0.0500 · 1k in / 500 out · 1 search · 1 X search · 2 code runs"
            " · 2 file searches · 1 MCP call · 1 X video call · 2 new tool calls · $1.50 today"
        )

    def test_append_pricing_embed_no_tool_usage(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, tool_usage={})
        assert embeds[0].description == "$0.0500 · 1k in / 500 out · $1.50 today"

    def test_append_generation_pricing_embed(self):
        from discord_grok.cogs.grok.embeds import append_generation_pricing_embed

        embeds: list[Embed] = []
        append_generation_pricing_embed(embeds, 0.07, 2.50, details=["1 image"])
        assert len(embeds) == 1
        assert embeds[0].description == "$0.0700 · 1 image · $2.50 today"
        assert embeds[0].colour == Colour(0)


class TestAppendReasoningEmbeds:
    """Tests for the append_reasoning_embeds helper."""

    def test_no_reasoning(self):
        from discord_grok.cogs.grok.embeds import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "")
        assert len(embeds) == 0

    def test_with_reasoning(self):
        from discord_grok.cogs.grok.embeds import append_reasoning_embeds

        embeds = []
        append_reasoning_embeds(embeds, "Some reasoning here")
        assert len(embeds) == 1
        assert embeds[0].title == "Reasoning"
        assert embeds[0].description == "||Some reasoning here||"

    def test_long_reasoning_truncated(self):
        from discord_grok.cogs.grok.embeds import append_reasoning_embeds

        embeds = []
        long_text = "a" * 4000
        append_reasoning_embeds(embeds, long_text)
        assert len(embeds) == 1
        assert len(embeds[0].description) < 3600
        assert "[reasoning truncated]" in embeds[0].description


class TestAppendResponseEmbeds:
    """Tests for the append_response_embeds helper."""

    def test_short_response(self):
        from discord_grok.cogs.grok.embeds import append_response_embeds

        embeds = []
        append_response_embeds(embeds, "Hello!")
        assert len(embeds) == 1
        assert embeds[0].title == "Response"
        assert embeds[0].description == "Hello!"

    def test_long_response_chunked(self):
        from discord_grok.cogs.grok.embeds import append_response_embeds

        embeds = []
        long_text = "a" * 7500
        append_response_embeds(embeds, long_text)
        assert len(embeds) > 1
        assert embeds[0].title == "Response"
        assert "Part" in embeds[1].title

    def test_very_long_response_preserved_for_delivery_batching(self):
        from discord_grok.cogs.grok.embeds import append_response_embeds

        embeds = []
        very_long_text = "a" * 25000
        append_response_embeds(embeds, very_long_text)
        total_text = "".join(embed.description for embed in embeds)
        assert total_text == very_long_text


class TestAppendSourcesEmbed:
    """Tests for the append_sources_embed helper."""

    def test_empty_citations_no_embed(self):
        from discord_grok.cogs.grok.embeds import append_sources_embed

        embeds = []
        append_sources_embed(embeds, [])
        assert len(embeds) == 0

    def test_web_citations_grouped(self):
        from discord_grok.cogs.grok.embeds import append_sources_embed

        citations = [
            {"url": "https://example.com/a", "source": "web"},
            {"url": "https://example.com/b", "source": "web"},
        ]
        embeds = []
        append_sources_embed(embeds, citations)
        assert len(embeds) == 1
        assert embeds[0].title == "Sources"
        assert "[example.com](https://example.com/a)" in embeds[0].description
        assert "[example.com](https://example.com/b)" in embeds[0].description

    def test_long_web_links_are_kept_complete_or_omitted(self):
        from discord_grok.cogs.grok.embeds import append_sources_embed

        first_url = "https://example.com/" + "a" * 3500
        second_url = "https://example.org/" + "b" * 1000
        embeds = []
        append_sources_embed(
            embeds,
            [
                {"url": first_url, "source": "web"},
                {"url": second_url, "source": "web"},
            ],
        )

        assert f"[example.com]({first_url})" in embeds[0].description
        assert second_url not in embeds[0].description
        assert len(embeds[0].description) <= 4000

    def test_mixed_sources_have_headings(self):
        from discord_grok.cogs.grok.embeds import append_sources_embed

        citations = [
            {"url": "https://example.com/a", "source": "web"},
            {"url": "https://x.com/i/status/123", "source": "x"},
        ]
        embeds = []
        append_sources_embed(embeds, citations)
        assert "**Web**" in embeds[0].description
        assert "**X Posts**" in embeds[0].description

    def test_single_source_type_no_heading(self):
        from discord_grok.cogs.grok.embeds import append_sources_embed

        citations = [
            {"url": "https://x.com/i/status/1", "source": "x"},
            {"url": "https://x.com/i/status/2", "source": "x"},
        ]
        embeds = []
        append_sources_embed(embeds, citations)
        assert "**X Posts**" not in embeds[0].description

    def test_skips_when_at_embed_limit(self):
        from discord_grok.cogs.grok.embeds import append_sources_embed

        embeds = [MagicMock() for _ in range(10)]
        citations = [{"url": "https://example.com", "source": "web"}]
        append_sources_embed(embeds, citations)
        assert len(embeds) == 10
