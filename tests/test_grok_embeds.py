from unittest.mock import MagicMock

from discord import Colour, Embed


class TestAppendPricingEmbed:
    """Tests for the append_pricing_embed helper."""

    def test_append_pricing_embed(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50)
        assert len(embeds) == 1
        desc = embeds[0].description
        assert "1,000 tokens in" in desc
        assert "500 tokens out" in desc
        assert "daily $1.50" in desc
        assert embeds[0].colour == Colour(0)

    def test_append_pricing_embed_with_reasoning_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, reasoning_tokens=200)
        assert len(embeds) == 1
        assert "200 reasoning" in embeds[0].description

    def test_append_pricing_embed_hides_zero_reasoning_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, reasoning_tokens=0)
        assert "reasoning" not in embeds[0].description

    def test_append_pricing_embed_with_cached_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, cached_tokens=300)
        assert "300 cached" in embeds[0].description

    def test_append_pricing_embed_with_image_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, image_tokens=200)
        assert "200 image" in embeds[0].description

    def test_append_pricing_embed_hides_zero_cached_and_image_tokens(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, cached_tokens=0, image_tokens=0)
        assert "cached" not in embeds[0].description
        assert "image" not in embeds[0].description

    def test_append_pricing_embed_with_tool_usage(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        tool_usage = {"SERVER_SIDE_TOOL_WEB_SEARCH": 3, "SERVER_SIDE_TOOL_X_SEARCH": 2}
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, tool_usage=tool_usage)
        desc = embeds[0].description
        assert desc is not None
        assert "Web Search ×3" in desc
        assert "X Search ×2" in desc
        assert "\n" in desc
        assert "tool cost" in desc

    def test_append_pricing_embed_no_tool_usage_line(self):
        from discord_grok.cogs.grok.embeds import append_pricing_embed

        embeds: list[Embed] = []
        append_pricing_embed(embeds, 0.05, 1000, 500, 1.50, tool_usage={})
        assert "\n" not in embeds[0].description

    def test_append_generation_pricing_embed(self):
        from discord_grok.cogs.grok.embeds import append_generation_pricing_embed

        embeds: list[Embed] = []
        append_generation_pricing_embed(embeds, 0.07, 2.50)
        assert len(embeds) == 1
        assert "$0.0700" in embeds[0].description
        assert "daily $2.50" in embeds[0].description


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
        assert "example.com" in embeds[0].description

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
