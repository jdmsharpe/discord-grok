from unittest.mock import MagicMock, patch

import pytest

from tests.support import make_cog


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

    async def test_resolve_selected_tools_includes_mcp(self, cog):
        from discord_grok.cogs.grok.tooling import McpServerConfig

        tools, error = cog.resolve_selected_tools(
            ["web_search", "mcp"],
            mcp_servers=[
                McpServerConfig(
                    server_url="https://mcp.example.com/sse",
                    server_label="mcp.example.com",
                    allowed_tool_names=["search"],
                )
            ],
        )

        assert error is None
        assert [tool["type"] for tool in tools] == ["web_search", "mcp"]
        assert tools[1]["server_url"] == "https://mcp.example.com/sse"
        assert tools[1]["allowed_tool_names"] == ["search"]


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

    def test_mcp_with_builtins(self):
        from discord_grok.cogs.grok.tooling import McpServerConfig, resolve_selected_tools

        tools, error = resolve_selected_tools(
            ["code_execution"],
            collection_ids=[],
            mcp_servers=[
                McpServerConfig(
                    server_url="https://mcp.example.com/sse",
                    server_label="mcp.example.com",
                    allowed_tool_names=["search"],
                )
            ],
        )

        assert error is None
        assert [tool["type"] for tool in tools] == ["code_interpreter", "mcp"]

    def test_resolve_tools_for_view_preserves_mcp_servers(self):
        from discord_grok.cogs.grok.models import (
            ChatCompletionParameters,
            Conversation,
            McpServerConfig,
        )
        from discord_grok.cogs.grok.state import resolve_tools_for_view

        conversation = Conversation(
            params=ChatCompletionParameters(
                model="grok-4.20",
                mcp_servers=[
                    McpServerConfig(
                        server_url="https://mcp.example.com/sse",
                        server_label="mcp.example.com",
                        allowed_tool_names=["search"],
                    )
                ],
            )
        )
        cog = make_cog(MagicMock())

        active_names, error = resolve_tools_for_view(cog, ["web_search"], conversation)

        assert error is None
        assert active_names == {"web_search"}
        assert [tool["type"] for tool in conversation.params.tools] == ["web_search", "mcp"]

    def test_every_ui_selectable_tool_is_resolvable_and_buildable(self):
        from discord_grok.cogs.grok.tooling import (
            SELECTABLE_TOOLS,
            TOOL_COLLECTIONS_SEARCH,
            TOOL_REGISTRY,
            resolve_selected_tools,
            resolve_tool_name,
        )

        for tool_name in SELECTABLE_TOOLS:
            assert tool_name in TOOL_REGISTRY
            tool_kwargs = {}
            collection_ids = []
            if tool_name == TOOL_COLLECTIONS_SEARCH:
                collection_ids = ["col_1"]
            elif TOOL_REGISTRY[tool_name].supports_kwargs:
                tool_kwargs = {"enabled": True}

            tools, error = resolve_selected_tools(
                [tool_name],
                collection_ids=collection_ids,
                x_search_kwargs=tool_kwargs if tool_name == "x_search" else None,
                web_search_kwargs=tool_kwargs if tool_name == "web_search" else None,
            )

            assert error is None
            assert len(tools) == 1
            assert resolve_tool_name(tools[0]) == tool_name
