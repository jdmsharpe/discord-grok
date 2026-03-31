from unittest.mock import patch

import pytest

from tests.grok_test_support import make_cog


class TestGrokCogTooling:
    """Tests for GrokCog tooling selection wrappers."""

    @pytest.fixture
    def cog(self, mock_bot):
        return make_cog(mock_bot)

    async def test_resolve_selected_tools_collections_requires_ids(self, cog):
        with patch("discord_grok.cogs.grok.tooling.XAI_COLLECTION_IDS", []):
            tools, error = cog.resolve_selected_tools(["collections_search"])

        assert tools == []
        assert "XAI_COLLECTION_IDS" in error

    async def test_resolve_selected_tools_success(self, cog):
        with patch("discord_grok.cogs.grok.tooling.XAI_COLLECTION_IDS", ["collection_abc"]):
            tools, error = cog.resolve_selected_tools(
                ["web_search", "x_search", "code_execution", "collections_search"]
            )

        assert error is None
        assert len(tools) == 4
        tool_types = sorted(tool["type"] for tool in tools)
        assert tool_types == [
            "code_interpreter",
            "file_search",
            "web_search",
            "x_search",
        ]

    async def test_resolve_selected_tools_x_search_with_kwargs(self, cog):
        """x_search kwargs should be included in the tool dict."""
        x_search_kw = {
            "enable_image_understanding": True,
            "allowed_x_handles": ["elonmusk"],
        }
        tools, error = cog.resolve_selected_tools(["x_search"], x_search_kwargs=x_search_kw)

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "x_search"
        assert tools[0]["enable_image_understanding"] is True
        assert tools[0]["allowed_x_handles"] == ["elonmusk"]

    async def test_resolve_selected_tools_x_search_without_kwargs(self, cog):
        """x_search without kwargs should still work."""
        tools, error = cog.resolve_selected_tools(["x_search"])

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "x_search"

    async def test_resolve_selected_tools_web_search_with_kwargs(self, cog):
        """web_search kwargs should be included in the tool dict."""
        web_search_kw = {
            "enable_image_understanding": True,
            "allowed_domains": ["example.com"],
        }
        tools, error = cog.resolve_selected_tools(["web_search"], web_search_kwargs=web_search_kw)

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search"
        assert tools[0]["enable_image_understanding"] is True
        assert tools[0]["allowed_domains"] == ["example.com"]

    async def test_resolve_selected_tools_web_search_without_kwargs(self, cog):
        """web_search without kwargs should still work."""
        tools, error = cog.resolve_selected_tools(["web_search"])

        assert error is None
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search"

    async def test_resolve_selected_tools_passes_date_strings(self, cog):
        """ISO date strings in x_search kwargs should be passed through."""
        x_search_kw = {
            "from_date": "2024-01-01T00:00:00",
            "to_date": "2024-12-31T00:00:00",
        }
        tools, error = cog.resolve_selected_tools(["x_search"], x_search_kwargs=x_search_kw)

        assert error is None
        assert tools[0]["from_date"] == "2024-01-01T00:00:00"
        assert tools[0]["to_date"] == "2024-12-31T00:00:00"


class TestResolveSelectedToolsUtil:
    """Tests for the resolve_selected_tools function in tooling.py."""

    def test_basic_resolution(self):
        from discord_grok.cogs.grok.tooling import resolve_selected_tools

        tools, error = resolve_selected_tools(
            ["web_search", "x_search", "code_execution"],
            collection_ids=[],
        )
        assert error is None
        assert len(tools) == 3

    def test_collections_requires_ids(self):
        from discord_grok.cogs.grok.tooling import resolve_selected_tools

        tools, error = resolve_selected_tools(
            ["collections_search"],
            collection_ids=[],
        )
        assert tools == []
        assert "XAI_COLLECTION_IDS" in error

    def test_collections_with_ids(self):
        from discord_grok.cogs.grok.tooling import resolve_selected_tools

        tools, error = resolve_selected_tools(
            ["collections_search"],
            collection_ids=["col_1"],
        )
        assert error is None
        assert tools[0]["type"] == "file_search"
        assert tools[0]["vector_store_ids"] == ["col_1"]
